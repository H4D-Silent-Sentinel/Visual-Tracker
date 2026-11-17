"""Screenshot every tab at desktop and phone width into runs/ui_check/shots/."""
from pathlib import Path
from playwright.sync_api import sync_playwright
O = Path(__file__).resolve().parent.parent / "runs/ui_check/shots"; O.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    for name, vp in [("d", {"width": 1440, "height": 900}), ("m", {"width": 390, "height": 844})]:
        pg = b.new_page(viewport=vp)
        pg.goto("http://localhost:8000/"); pg.wait_for_function("typeof R!=='undefined' && R.clip && R.pred", timeout=20000)
        pg.locator("#demoList .crow").first.click(); pg.wait_for_timeout(1200); pg.evaluate("seekFrame(372)"); pg.wait_for_timeout(800)
        pg.screenshot(path=str(O / f"review_{name}.png"), full_page=(name == "m"))
        for t in ["live", "results", "pool", "collect"]:
            pg.locator(f"nav button[data-tab={t}]").click(); pg.wait_for_timeout(4000 if t == "live" else 1800)
            pg.screenshot(path=str(O / f"{t}_{name}.png"), full_page=(t != "live" and name == "d") or name == "m")
        pg.close()
    b.close()
print("ok")
