import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest


class LauncherTests(unittest.TestCase):
    def exercise(self, case):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            launcher = root / 'Start Tracker.command'
            launcher.write_text((Path(__file__).resolve().parents[1] / 'Start Tracker.command').read_text())
            bindir = root / 'bin'
            bindir.mkdir()
            (root / '.venv/bin').mkdir(parents=True)
            common = f'#!{Path(sys.executable).resolve()}\nimport os, sys, time, json\nfrom pathlib import Path\nr=Path(os.environ["CHECK_ROOT"])\n'
            programs = {
                root / '.venv/bin/python': common + '''
with (r/'children').open('a') as f: f.write(json.dumps({'pid':os.getpid(),'args':sys.argv[1:],'cache':os.environ.get('HF_HOME')})+'\\n')
if os.environ['CHECK_CASE']=='failed': sys.exit(1)
while True: time.sleep(.1)
''',
                bindir / 'curl': common + '''
url=sys.argv[-1]
with (r/'requests').open('a') as f: f.write(url+'\\n')
if '/live/set?' in url or os.environ['CHECK_CASE']=='reuse': sys.exit(0)
if os.environ['CHECK_CASE']=='failed': sys.exit(7)
script='vlm_server.py' if '/health' in url else 'dashboard/app.py'
if (r/'children').exists():
    for line in (r/'children').read_text().splitlines():
        child=json.loads(line)
        if script in child['args']:
            try: os.kill(child['pid'],0); sys.exit(0)
            except ProcessLookupError: pass
sys.exit(7)
''',
                bindir / 'open': common + "(r/'opened').write_text(sys.argv[-1])\n",
                bindir / 'sleep': common + 'time.sleep(.01)\n',
            }
            for path, text in programs.items():
                path.write_text(text)
                path.chmod(0o755)
            env = {**os.environ, 'PATH': str(bindir)+os.pathsep+os.environ['PATH'],
                   'CHECK_ROOT': folder, 'CHECK_CASE': case}
            process = subprocess.Popen(['zsh', str(launcher)], cwd='/tmp', env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            output = ''
            try:
                deadline = time.monotonic()+5
                while time.monotonic()<deadline and process.poll() is None:
                    if (root/'opened').exists():
                        break
                    if select.select([process.stdout], [], [], .05)[0]:
                        output += process.stdout.readline()
                opened = (root/'opened').read_text() if (root/'opened').exists() else None
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                output += process.communicate(timeout=5)[0]
            children = [json.loads(line) for line in (root/'children').read_text().splitlines()] if (root/'children').exists() else []
            for child in children:
                with self.assertRaises(ProcessLookupError):
                    os.kill(child['pid'], 0)
                self.assertEqual(child['cache'], str(root/'.venv/huggingface'))
            requests = (root/'requests').read_text() if (root/'requests').exists() else ''
            return opened, children, requests, output, process.returncode

    def test_starts_both_services_enables_verification_and_cleans_up(self):
        opened, children, requests, _, _ = self.exercise('owned')
        self.assertEqual(opened, 'http://localhost:8000/#live')
        self.assertEqual(len(children), 2)
        self.assertIn('/live/set?verify=true&paused=false', requests)

    def test_reuses_existing_services_without_claiming_their_processes(self):
        opened, children, _, _, _ = self.exercise('reuse')
        self.assertEqual(opened, 'http://localhost:8000/#live')
        self.assertEqual(children, [])

    def test_failed_start_does_not_open_browser(self):
        opened, _, _, output, code = self.exercise('failed')
        self.assertIsNone(opened)
        self.assertIn('Service stopped before it was ready', output)
        self.assertNotEqual(code, 0)


if __name__ == '__main__':
    unittest.main()
