"""Browser check of the Collect tab (run against a test server: PORT=8001).
Run: .venv/bin/python tests/browser/collect_test.py [http://localhost:8001]
"""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
O = Path(__file__).resolve().parents[2] / "runs/ui_check"
O.mkdir(parents=True, exist_ok=True)
ok = []
def check(n, c, d=""): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"  [{d}]" if d else ""), flush=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True); errs = []
    pg = b.new_page(viewport={"width": 1440, "height": 900}); pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL + "/#collect"); pg.wait_for_function("typeof C!=='undefined' && C.clip && document.getElementById('labVid').readyState>=1", timeout=30000); pg.wait_for_timeout(1500)
    cams = pg.locator("#camIdx option").all_inner_texts(); check("camera menu lists cameras by name", any("phone" in c or "MacBook" in c for c in cams), "; ".join(cams))
    groups = pg.evaluate("[...document.querySelectorAll('#labClip optgroup')].map(g=>g.label)"); check("labeler offers only clips that need labels", not any("Tracking" in g for g in groups), str(groups))
    pg.select_option("#labClip", "demo__makers_party_uav"); pg.wait_for_function("C.clip && C.clip.key==='demo__makers_party_uav' && document.getElementById('labVid').readyState>=1", timeout=20000); pg.wait_for_timeout(1500)
    clip = pg.evaluate("C.clip.key")
    state = pg.evaluate("({frames:C.clip.frames,saved:Object.keys(C.saved).map(Number)})")
    f0 = next((frame for frame in range(0, state["frames"] - 5, 5) if frame not in state["saved"] and frame + 5 not in state["saved"]), None)
    if f0 is None: raise RuntimeError("No unlabeled frame pair available; existing user labels were left untouched.")
    pg.locator('#labAuto').uncheck(); pg.evaluate(f'labGo({f0})')
    st = pg.locator("#labStatus").inner_text(); print("   status on load:", st)
    pg.locator("#labBox").scroll_into_view_if_needed(); pg.wait_for_timeout(300)
    cv = pg.locator("#labCv").bounding_box()
    pg.mouse.move(cv["x"] + cv["width"] * .45, cv["y"] + cv["height"] * .40); pg.mouse.down(); pg.mouse.move(cv["x"] + cv["width"] * .52, cv["y"] + cv["height"] * .48, steps=4); pg.mouse.up()
    pg.select_option('#labClass', 'drone'); pg.evaluate('document.activeElement.blur()')
    pg.keyboard.press("s"); pg.wait_for_timeout(900)
    f1 = pg.evaluate("C.frame"); saved = pg.evaluate("Object.keys(C.saved)")
    check("S saves the drawn box and moves on by the step", str(f0) in saved and f1 == f0 + 5, f"frame {f0} -> {f1}")
    pg.keyboard.press("n"); pg.wait_for_timeout(900)
    lab = pg.evaluate(f"fetch('/api/labels?clip={clip}').then(r=>r.json()).then(d=>d.classifications)")
    check("N saves explicit background and the manual class survives", lab.get(str(f0), {}).get('label') == 'drone' and lab.get(str(f1), {}).get('no_drone') is True and lab.get(str(f1), {}).get('boxes') == [], json.dumps(lab)[:120])
    pg.screenshot(path=str(O / "collect_new_d.png"), full_page=True)
    pg.locator("#labStrip").click(position={"x": 5, "y": 9}); pg.wait_for_timeout(500)
    check("clicking the progress strip jumps", pg.evaluate("C.frame") < 15, str(pg.evaluate("C.frame")))
    pg.evaluate("collectLoad()"); pg.wait_for_timeout(1000)
    ds = pg.locator("#classificationSummary").inner_text(); check("classification summary counts manual frames", "manual frames" in ds, ds)
    pg.locator('#collectionName').fill(f'browser_collect_{time.time_ns()}')
    pg.locator("#collectionExport").click(); pg.wait_for_timeout(800)
    msg = pg.locator("#collectionMsg").inner_text(); check("Export test collection writes frozen manifests", "test.jsonl" in msg and "development.jsonl" in msg, msg[:120])
    # clean up the test labels
    for f in (f0, f0 + 5):
        if str(f) in lab: pg.evaluate(f"fetch('/api/classification/label',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{clip:'{clip}',frame:{f},delete:true}})}})")
    m = b.new_page(viewport={"width": 390, "height": 844}); m.on("pageerror", lambda e: errs.append(str(e)))
    m.goto(URL + "/#collect"); m.wait_for_timeout(3000)
    ov = m.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth"); check("phone width: no sideways scroll", ov <= 1, f"{ov}px")
    m.screenshot(path=str(O / "collect_new_m.png"), full_page=True)
    check("no JavaScript errors", not errs, "; ".join(errs)[:200])
    b.close()
print(f"{sum(ok)} of {len(ok)} passed")
