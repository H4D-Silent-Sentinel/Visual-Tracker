"""Record a captioned demo run of the dashboard to ../../demo/dashboard_demo.mp4.
Run against a dashboard server:  .venv/bin/python dashboard/record_demo.py http://localhost:8001
"""
import shutil, subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
OUT = Path(__file__).resolve().parents[1] / "runs" / "demo"
TMP = OUT / "_raw"
W, H = 1600, 900

CAP_JS = """(t) => { let c = document.getElementById('demoCap');
  if (!c) { c = document.createElement('div'); c.id = 'demoCap';
    c.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:9999;background:rgba(10,14,22,.92);color:#fff;'
      + 'font:600 22px system-ui,-apple-system,sans-serif;padding:12px 22px;border-radius:10px;border:1px solid rgba(255,255,255,.18);'
      + 'box-shadow:0 8px 30px rgba(0,0,0,.45);max-width:80%;text-align:center;transition:opacity .25s';
    document.body.appendChild(c); }
  c.style.opacity = t ? 1 : 0; if (t) c.textContent = t; }"""


def run():
    shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(TMP), record_video_size={"width": W, "height": H})
        pg = ctx.new_page()
        cap = lambda t: pg.evaluate(CAP_JS, t)
        wait = pg.wait_for_timeout
        tab = lambda t: pg.locator(f"nav button[data-tab={t}]").click()

        # 0. title
        pg.goto(URL + "/#review"); pg.wait_for_function("typeof R!=='undefined' && R.clip", timeout=30000)
        tab("live"); pg.wait_for_function("document.getElementById('lvC').textContent.includes('%')", timeout=60000)
        pg.select_option("#lvSource", "footage/dut/video09.mp4"); wait(2500)
        cap("CENTCOM-4 Drone Hide and Seek · visual drone tracker demo"); wait(3500)

        # 1. live tracking
        cap("Live: the model runs on every frame and locks onto the drone"); wait(6000)
        cap("Raise the confidence cutoff and weak boxes drop away"); wait(800)
        for v in ["0.5", "0.65", "0.8", "0.9"]:
            pg.locator("#lvConf").fill(v); wait(700)
        wait(1800)
        pg.locator("#lvConf").fill("0.35"); wait(1200)
        cap("Drone crossing in front of trees, the hard case"); pg.select_option("#lvSource", "footage/dut/video17.mp4"); wait(6500)

        # 2. clip review
        tab("review"); wait(800)
        pg.evaluate("openClip('dut__video17', 330, 'v1')"); wait(1500)
        cap("Clip Review: same clip, scrubbable frame by frame, true drone box in white"); wait(1500)
        pg.evaluate("document.getElementById('vid').playbackRate = 0.5; document.getElementById('vid').play()"); wait(2600)
        # pick a frame in the tree crossing where v1 misses and some other output holds the drone
        f = pg.evaluate("""async () => {
          const st = {};
          for (const o of ['v1','v2','v3','v2_hold']) { await setOutput(o, true); st[o] = Array.from(R.st); }
          for (let k = 340; k < 460; k++) if (st.v1[k] === 4 && (st.v2_hold[k] === 3 || st.v2[k] === 3)) return k;
          return 372; }""")
        pg.evaluate(f"setOutput('v1'); document.getElementById('vid').pause(); seekFrame({f})"); wait(900)
        cap("Paused on a miss: red on the timeline, v1 has no box on the drone"); wait(3200)
        word = lambda: pg.evaluate("({3:'box on the drone',4:'misses it',5:'false alarm'})[R.st[R.frame]] || 'no box'")
        for o, t in [("v2", "v2"), ("v3", "v3"), ("v2_hold", "v2 + hold tracker")]:
            pg.evaluate(f"setOutput('{o}')"); wait(300); cap(f"Same frame, {t}: {word()}"); wait(2200)
        # birds: a frame where v1 boxes a bird and v3 draws nothing
        k = pg.evaluate("""async () => {
          await openClip('negatives__terns_midway', 0, 'v1');
          const a = await (await fetch('/api/pred?clip=negatives__terns_midway&out=v1')).json();
          const b = await (await fetch('/api/pred?clip=negatives__terns_midway&out=v3')).json();
          let best = null, conf = 0;
          a.boxes.forEach((bs, i) => { const c = bs.length ? Math.max(...bs.map(x => x[4])) : 0; if (c > conf && !(b.boxes[i] || []).length) { conf = c; best = i } });
          return best; }""")
        pg.evaluate(f"openClip('negatives__terns_midway', {k or 0}, 'v1')"); wait(1500)
        cap("Birds, no drone in this clip: v1 calls a tern a drone"); wait(3000)
        pg.evaluate("setOutput('v3')"); wait(300); cap("Same frame, v3: no box, no false alarm"); wait(3000)

        # 3. results
        tab("results"); wait(1200)
        cap("Results: every version scored on the same held-out footage"); wait(3000)
        for _ in range(14):
            pg.mouse.wheel(0, 140); wait(220)
        wait(1500)

        # 4. data pool
        tab("pool"); pg.evaluate("scrollTo(0,0)"); wait(1500)
        cap("Data Pool: every photo and video we train and test on"); wait(2500)
        pg.locator("#imgs .th").nth(3).click(); wait(2200); pg.locator("#boxClose").click(); wait(600)
        for _ in range(10):
            pg.mouse.wheel(0, 160); wait(200)
        wait(1200)

        # 5. collect
        tab("collect"); pg.evaluate("scrollTo(0,0)"); wait(1200)
        pg.select_option("#labClip", "demo__makers_party_uav")
        pg.wait_for_function("C.clip && C.clip.key==='demo__makers_party_uav' && document.getElementById('labVid').readyState>=2", timeout=20000)
        pg.evaluate("labGo(400)"); wait(2500)
        cap("Collect: record our own footage, label it, and build the next training set"); wait(3000)
        for _ in range(8):
            pg.mouse.wheel(0, 150); wait(220)
        wait(2000)
        cap("Next: rooftop footage over trees, labeled here, then retrain"); wait(3500)
        cap(""); wait(600)
        vid = pg.video.path()
        ctx.close(); b.close()
    dst = OUT / "dashboard_demo.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-r", "30", "-movflags", "+faststart", str(dst)], check=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print("wrote", dst)


if __name__ == "__main__":
    run()
