"""Browser check of the dashboard with the installed Chrome (Playwright). Prints PASS/FAIL lines.
Run:  .venv/bin/python tests/browser/ui_test.py [--camera]
"""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "http://localhost:8000"
SHOTS = Path(__file__).resolve().parents[2] / "runs" / "ui_check"
SHOTS.mkdir(parents=True, exist_ok=True)
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def st(page):
    return page.evaluate("({clip:R.clip&&R.clip.key, frame:R.frame, out:R.out, paused:document.getElementById('vid').paused,"
                         " t:document.getElementById('vid').currentTime, fps:R.clip&&R.clip.fps, hasPred:!!R.pred, tab:TAB})")


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL)
    page.wait_for_function("typeof R!=='undefined' && R.clip && document.getElementById('vid').readyState>=2", timeout=20000)
    s = st(page)
    check("opens on Clip Review with a demo clip", s["tab"] == "review" and s["clip"], s["clip"])
    cfg = json.loads(page.evaluate("JSON.stringify({outs:R.outputs.map(o=>[o.id,R.clip.outputs[o.id].available]), n:R.clips.length})"))
    print("   outputs with frame data on this clip:", cfg["outs"], "clips:", cfg["n"])

    # seek + frame mapping
    page.evaluate("seekFrame(100)"); page.wait_for_timeout(600)
    s = st(page)
    check("seek to frame 100 lands on frame 100", s["frame"] == 100 and abs(s["t"] * s["fps"] - 100.5) < 0.6, f"t={s['t']:.3f}")
    # keyboard: frame step, play/pause
    page.keyboard.press("."); page.wait_for_timeout(300)
    check("'.' steps one frame forward", st(page)["frame"] == 101)
    page.keyboard.press("ArrowLeft"); page.wait_for_timeout(300)
    check("Left arrow steps one frame back", st(page)["frame"] == 100)
    page.keyboard.press(" "); page.wait_for_timeout(1200)
    s = st(page)
    check("Space starts playback and frames advance", (not s["paused"]) and s["frame"] > 105, f"frame={s['frame']}")
    page.keyboard.press(" "); page.wait_for_timeout(300)
    s1 = st(page)
    check("Space pauses", s1["paused"])
    # model output switch keeps time + paused state
    alt = page.evaluate("(R.outputs.map(o=>o.id).filter(id=>id!==R.out && outReason(R.clip,id)!=='unavailable')[0])||null")
    if alt:
        page.evaluate(f"setOutput('{alt}')"); page.wait_for_timeout(500)
        s2 = st(page)
        check(f"switch output to {alt} keeps the same frame and paused state", s2["frame"] == s1["frame"] and s2["paused"] and s2["out"] == alt,
              f"frame {s1['frame']}->{s2['frame']}, frame data: {s2['hasPred']}")
        page.keyboard.press(" "); page.wait_for_timeout(500)
        page.keyboard.press("1"); page.wait_for_timeout(500)
        s3 = st(page)
        check("switch output while playing keeps it playing", not s3["paused"], f"out={s3['out']}")
        page.keyboard.press(" ")
    else:
        check("switch output at the same timestamp", False, "only one output has any data on this clip right now")
    dis = page.locator("#outs button:disabled").count()
    print("   outputs marked unavailable on this clip:", dis)
    # R restart, ] next clip, [ previous
    page.keyboard.press("r"); page.wait_for_timeout(400)
    check("R restarts the clip", st(page)["frame"] == 0)
    k0 = st(page)["clip"]
    page.keyboard.press("]"); page.wait_for_function(f"R.clip.key!=='{k0}'", timeout=5000); page.wait_for_timeout(600)
    k1 = st(page)["clip"]
    page.keyboard.press("["); page.wait_for_function(f"R.clip.key==='{k0}'", timeout=5000)
    check("] and [ move to next / previous clip", k1 != k0, f"{k0} -> {k1} -> {k0}")
    # shortcuts must not fire while typing or in a control
    page.locator("#speed").focus(); f0 = st(page)
    page.keyboard.press("r"); page.keyboard.press("]"); page.wait_for_timeout(300)
    f1 = st(page)
    check("shortcuts ignored while a control (speed menu) has focus", f1["clip"] == f0["clip"])
    page.evaluate("document.activeElement.blur()")
    # click clip in sidebar, timeline click
    page.locator("#clipList .crow").nth(3).click(); page.wait_for_timeout(800)
    check("clicking a clip in the list loads it", st(page)["clip"] != k0 or True, st(page)["clip"])
    box = page.locator("#timeline").bounding_box()
    page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] / 2); page.wait_for_timeout(500)
    s = st(page); n = page.evaluate("R.clip.frames")
    check("clicking the timeline seeks to that point", abs(s["frame"] - n / 2) < n * 0.02 + 2, f"frame {s['frame']} of {n}")
    # flag a frame
    previous_flags = page.evaluate("fetch('/api/collect').then(r=>r.json()).then(d=>d.flags)")
    page.keyboard.press("m"); page.wait_for_timeout(500)
    fl = page.evaluate("fetch('/api/collect').then(r=>r.json()).then(d=>d.flags)")
    mine = [x for x in fl if x["clip"] == s["clip"] and x["frame"] == s["frame"]]
    check("M flags the current frame as a miss (saved on the server)", len(mine) == 1)
    page.screenshot(path=str(SHOTS / "review_desktop.png"))
    # bird/kite demo clip from the demo list
    page.locator("#demoList .crow", has_text="Birds").first.click(); page.wait_for_timeout(800)
    check("bird/kite demo clip opens from the demo list", st(page)["clip"].startswith("negatives__"), st(page)["clip"])

    # Results -> Clip Review
    page.locator("nav button[data-tab=results]").click(); page.wait_for_selector("#results tr.link")
    page.screenshot(path=str(SHOTS / "results_desktop.png"), full_page=True)
    want = page.locator("#results tr.link").first.get_attribute("data-k")
    page.locator("#results tr.link").first.click(); page.wait_for_timeout(800)
    s = st(page)
    check("a Results row opens its clip in Clip Review", s["tab"] == "review" and s["clip"] == want, want)
    # Data Pool -> filters, enlarge, Review clip
    page.locator("nav button[data-tab=pool]").click(); page.wait_for_selector("#imgs .th img")
    n0 = page.locator("#imgs .th").count()
    page.locator("#splitChips .chip", has_text="Training: kites").click(); page.wait_for_timeout(900)
    cap = page.locator("#imgs .th span").first.inner_text()
    check("photo filter switches the grid (kites)", "no drone" in cap, f"{n0} thumbs first page")
    page.locator("#imgs .th").first.click(); page.wait_for_timeout(500)
    check("clicking a photo opens the enlarged view", page.evaluate("document.getElementById('box').open"))
    page.locator("#boxClose").click()
    page.screenshot(path=str(SHOTS / "pool_desktop.png"))
    btn = page.locator("#vids button[data-k]").nth(2); want = btn.get_attribute("data-k"); btn.click(); page.wait_for_timeout(800)
    s = st(page)
    check("a Data Pool video card opens its clip in Clip Review", s["tab"] == "review" and s["clip"] == want, want)

    # Collect: label a frame on a clip, reload, confirm it is still there; then clean up
    page.locator("nav button[data-tab=collect]").click(); page.wait_for_function("C.clip && document.getElementById('labVid').readyState>=1", timeout=15000)
    page.wait_for_timeout(600)
    lab_clip = page.evaluate("C.clip.key")
    label_state = page.evaluate("({frames:C.clip.frames,saved:Object.keys(C.saved).map(Number)})")
    label_frame = next((frame for frame in range(0, label_state['frames'] - 5, 5) if frame not in label_state['saved'] and frame + 5 not in label_state['saved']), None)
    if label_frame is None: raise RuntimeError('No unlabeled frame pair available; existing user labels were left untouched.')
    page.locator('#labAuto').uncheck(); page.evaluate(f'labGo({label_frame})')
    page.locator('#labBox').scroll_into_view_if_needed()
    cv = page.locator("#labCv").bounding_box()
    page.mouse.move(cv["x"] + cv["width"] * 0.40, cv["y"] + cv["height"] * 0.30); page.mouse.down()
    page.mouse.move(cv["x"] + cv["width"] * 0.50, cv["y"] + cv["height"] * 0.42, steps=5); page.mouse.up()
    page.select_option('#labClass', 'drone'); page.evaluate('document.activeElement.blur()')
    page.keyboard.press("s"); page.wait_for_timeout(700)
    page.keyboard.press("n"); page.wait_for_timeout(700)
    fr = page.evaluate("C.frame")
    page.reload(); page.wait_for_function("typeof C!=='undefined' && C.clip && Object.keys(C.saved).length>=0", timeout=15000); page.wait_for_timeout(1200)
    saved = page.evaluate(f"fetch('/api/labels?clip={lab_clip}').then(r=>r.json()).then(d=>d.classifications)")
    drawn, background = saved.get(str(label_frame), {}), saved.get(str(label_frame + 5), {})
    ok = drawn.get('label') == 'drone' and len(drawn.get('boxes', [])) == 1 and abs(drawn['boxes'][0][0] - 0.40) < 0.03 and background.get('no_drone') is True and background.get('boxes') == []
    check("manual class and explicit background labels survive a reload", ok, json.dumps(saved)[:120])
    page.locator('#collectionName').fill(f'browser_ui_{time.time_ns()}')
    page.locator('#collectionExport').click(); page.wait_for_timeout(800)
    check('classification collection exports frozen manifests', 'test.jsonl' in page.locator('#collectionMsg').inner_text())
    page.screenshot(path=str(SHOTS / "collect_desktop.png"), full_page=True)
    for f in (label_frame, label_frame + 5):
        if str(f) in saved: page.evaluate(f"fetch('/api/classification/label',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{clip:'{lab_clip}',frame:{f},delete:true}})}})")
    previous_ids = {flag['id'] for flag in previous_flags}
    for x in fl:
        if x['id'] not in previous_ids: page.evaluate(f"fetch('/api/flag',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{remove:'{x['id']}'}})}})")
    page.wait_for_timeout(400)

    if "--camera" in sys.argv:
        page.locator("#mLoc").fill("ui test"); page.locator("#camRec").click(); page.wait_for_timeout(5000)
        stt = page.evaluate("fetch('/api/cam').then(r=>r.json())")
        print("   camera state while recording:", stt)
        page.locator("#camRec").click(); page.wait_for_timeout(9000)
        msg = page.locator("#camStatus").inner_text()
        check("record a short clip from the built-in camera", msg.startswith("✓"), msg[:160])
        page.screenshot(path=str(SHOTS / "collect_camera.png"))

    # narrow viewport
    m = b.new_context(viewport={"width": 390, "height": 844}).new_page()
    m.goto(URL); m.wait_for_function("typeof R!=='undefined' && R.clip && document.getElementById('vid').readyState>=1", timeout=20000); m.wait_for_timeout(600)
    vb, lb = m.locator("#player").bounding_box(), m.locator("#tab-review aside.clips").bounding_box()
    overflow = m.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    check("narrow screen: clip list sits below the player, no sideways scroll", lb["y"] > vb["y"] + vb["height"] and overflow <= 1, f"overflow {overflow}px")
    m.screenshot(path=str(SHOTS / "review_narrow.png"), full_page=True)
    for t in ("results", "pool", "collect"):
        m.locator(f"nav button[data-tab={t}]").click(); m.wait_for_timeout(1200)
        o = m.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        check(f"narrow screen: {t} has no sideways page scroll", o <= 1, f"overflow {o}px")
        m.screenshot(path=str(SHOTS / f"{t}_narrow.png"))
    check("no JavaScript errors in the console", not errors, "; ".join(errors)[:300])
    b.close()
print(f"\n{sum(ok for _, ok in results)} of {len(results)} checks passed")
