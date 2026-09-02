#!/usr/bin/env python
"""Drive a Foxglove-protocol viewer against lerobot's playback server and record it.

Serves: `lerobot-dataset-viz --display-mode foxglove` (the PR-4542 path, which works on
lance roots including s3://). Views it in Lichtblick (the open-source Foxglove Studio
fork, same Foxglove WebSocket protocol) so the capture needs no hosted account.
Records the viewport to PNG frames, then stitches an mp4/gif with ffmpeg.
"""
import os
import argparse, json, os, signal, subprocess, sys, time, glob
from pathlib import Path


def rx_bytes():
    tot = 0
    for p in glob.glob("/sys/class/net/*/statistics/rx_bytes"):
        if "/lo/" in p:
            continue
        try: tot += int(Path(p).read_text())
        except Exception: pass
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")
    ap.add_argument("--repo-id", default="lerobot/droid_1.0.1")
    ap.add_argument("--episode-index", type=int, default=7)
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--viewer", default="http://127.0.0.1:8080")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--region", default=None)
    ap.add_argument("--outdir", default="out/assets")
    ap.add_argument("--name", default="foxglove_droid")
    ap.add_argument("--server-cmd", default=None, help="run this instead of lerobot-dataset-viz")
    ap.add_argument("--layout", default="fox_layout", help="module exposing build()")
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    frames = out / f"{args.name}_frames"; frames.mkdir(exist_ok=True)
    for f in frames.glob("*.png"): f.unlink()

    env = dict(os.environ)
    if args.region: env["AWS_DEFAULT_REGION"] = args.region
    srvlog = out / f"{args.name}_server.log"
    if args.server_cmd:
        cmd = args.server_cmd.split()
    else:
        cmd = ["lerobot-dataset-viz", "--repo-id", args.repo_id, "--root", args.root,
               "--episode-index", str(args.episode_index), "--display-mode", "foxglove",
               "--host", "127.0.0.1", "--web-port", str(args.ws_port), "--tolerance-s", "0.001"]
    print("server:", " ".join(cmd), flush=True)
    with open(srvlog, "w") as lf:
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 900:
        if srvlog.exists() and "Started server" in srvlog.read_text(): break
        if srv.poll() is not None:
            print("server died:\n", srvlog.read_text()[-1500:]); return 1
        time.sleep(1)
    print(f"server up in {time.perf_counter()-t0:.1f}s", flush=True)

    # Foxglove/Lichtblick deep-link: auto-open the foxglove-websocket source
    url = (f"{args.viewer}/?ds=foxglove-websocket"
           f"&ds.url=ws%3A%2F%2F127.0.0.1%3A{args.ws_port}")
    print("viewer:", url, flush=True)

    b0, tb = rx_bytes(), time.perf_counter()
    from playwright.sync_api import sync_playwright
    n = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage",
                                          "--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page(viewport={"width": 1600, "height": 900},
                                device_scale_factor=1)
        # Lichtblick's index.html itself sets
        #   globalThis.LICHTBLICK_SUITE_DEFAULT_LAYOUT = [][0];
        # in an inline script, which runs AFTER add_init_script and clobbers it.
        # So rewrite that literal in the HTML on the way through instead.
        sys.path.insert(0, ".")
        build_layout = __import__(args.layout, fromlist=['build']).build
        layout_json = json.dumps(build_layout())

        injected = {"done": False}

        def _inject(route):
            if injected["done"] or route.request.resource_type != "document":
                return route.continue_()
            resp = route.fetch()
            body = resp.text()
            marker = "globalThis.LICHTBLICK_SUITE_DEFAULT_LAYOUT = [][0];"
            if marker in body:
                body = body.replace(
                    marker,
                    "globalThis.LICHTBLICK_SUITE_DEFAULT_LAYOUT = " + layout_json + ";")
                injected["done"] = True
                print("  injected custom layout into index.html", flush=True)
            else:
                print("  WARNING: layout marker not found; default layout will be used",
                      flush=True)
            hdrs = {k: v for k, v in resp.headers.items() if k.lower() != "content-length"}
            route.fulfill(status=resp.status, headers=hdrs, body=body)

        page.route("**/*", _inject)
        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(20000)          # let it connect + lay out panels
        # close the panel-settings sidebar so the capture is all data
        try:
            page.get_by_role("button", name="Close").first.click(timeout=4000)
        except Exception:
            try: page.mouse.click(305, 58)
            except Exception: pass
        page.wait_for_timeout(2500)
        page.screenshot(path=str(out / f"{args.name}_connected.png"))
        interval = 1.0 / args.fps
        end = time.perf_counter() + args.seconds
        while time.perf_counter() < end:
            page.screenshot(path=str(frames / f"f{n:05d}.png"))
            n += 1
            time.sleep(max(0.0, interval - 0.15))
        page.screenshot(path=str(out / f"{args.name}_final.png"))
        browser.close()
    mb = (rx_bytes() - b0) / 1e6
    watched = time.perf_counter() - tb

    srv.send_signal(signal.SIGINT)
    try: srv.wait(timeout=30)
    except subprocess.TimeoutExpired: srv.kill()

    mp4 = out / f"{args.name}.mp4"
    if n:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", str(frames / "f%05d.png"), "-vf",
                        "scale=1280:-2:flags=lanczos", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "26", str(mp4)], check=False)
    res = {"root": args.root, "episode_index": args.episode_index, "frames_captured": n,
           "watched_s": round(watched, 1), "object_store_MB_pulled": round(mb, 2),
           "mp4": str(mp4) if mp4.exists() else None}
    print(json.dumps(res, indent=2), flush=True)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
