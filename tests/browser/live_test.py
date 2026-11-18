"""Quick browser check of the Live tab.
Run: .venv/bin/python tests/browser/live_test.py
"""
from pathlib import Path
from playwright.sync_api import sync_playwright
S = Path(__file__).resolve().parents[2] / "runs" / "ui_check"; S.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    pg = b.new_page(viewport={"width": 1440, "height": 900}); errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000/#live")
    pg.wait_for_function("document.getElementById('lvC').textContent.includes('%')", timeout=60000)
    pg.wait_for_timeout(2500)
    a = pg.evaluate("fetch('/live/stats').then(r=>r.json())")
    print("status", a["status"], "fps", a["fps"], "model", a["settings"]["model"], "conf", a["settings"]["conf"], "frames", a["frames"])
    pg.locator("#lvConf").fill("0.9"); pg.wait_for_timeout(1500)
    c = pg.evaluate("fetch('/live/stats').then(r=>r.json())")
    print("after slider -> cutoff", c["settings"]["conf"], "boxes in view", len(c["targets"]))
    pg.locator("#lvConf").fill("0.35"); pg.keyboard.press("Tab")
    pg.evaluate("document.activeElement.blur()"); pg.keyboard.press(" "); pg.wait_for_timeout(700)
    d = pg.evaluate("fetch('/live/stats').then(r=>r.json())"); f1 = d["frames"]; pg.wait_for_timeout(700)
    e = pg.evaluate("fetch('/live/stats').then(r=>r.json())")
    print("space pauses:", e["settings"]["paused"] and e["frames"] == f1)
    pg.keyboard.press(" "); pg.keyboard.press("]"); pg.wait_for_timeout(2500)
    f = pg.evaluate("fetch('/live/stats').then(r=>r.json())")
    print("] next clip:", f["settings"]["source"], f["status"])
    pg.screenshot(path=str(S / "live_desktop.png"))
    pg.locator("nav button[data-tab=review]").click(); pg.wait_for_timeout(1500)
    g = pg.evaluate("fetch('/live/stats').then(r=>r.json())"); pg.wait_for_timeout(800)
    h = pg.evaluate("fetch('/live/stats').then(r=>r.json())")
    print("engine idles after leaving the tab:", h["frames"] == g["frames"] or h["status"] == "IDLE", "| js errors:", errs)
    b.close()
