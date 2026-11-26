"""Hand Mouse: move the Windows cursor with your hand through the webcam.

The tracking runs in a separate helper process (its own venv with mediapipe + opencv,
under api.local_dir()), so the app's Python never loads those libraries. The helper
moves the real cursor, so it drives Ducky, UEFN and every other app the same way.

All runtime state lives in files in that folder (setup.json, helper.pid, status.json),
never in this module's memory: the panel and the agent's tools can run in different
copies of this backend and must see the same thing.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import helper_src

DEFAULTS = {
    "camera": 0,          # webcam index
    "smoothing": 0.5,     # steadiness at rest (One Euro filter); fast moves stay responsive
    "margin": 0.15,       # frame border ignored, so the screen edge is reachable
    "pinch": 0.35,        # pinch distance / hand size that counts as a click
    "scroll_speed": 1,    # wheel notches per step
    "overlay": True,      # ring around the cursor
    "palm_hold": 1.0,     # seconds of open palm to pause / resume
    "drag_hold": 0.35,    # a pinch held this long starts a drag; a quicker pinch is a click
    "two_hands": True,    # both hands in view: one moves the cursor, the other clicks
    "pointer_hand": "right",
    "game_mode": False,   # stick hand = direction keys, look hand = camera (Ctrl+Alt+G toggles)
    "look_speed": 4.0,    # game camera speed
    "stick_dead": 0.35,   # game stick deadzone, in hand sizes
    "predict": 0.5,       # how far the cursor leads a moving hand (0 = none, 1 = the full camera delay)
    "fast_camera": True,  # ask a real webcam for 60 fps (virtual cameras are left alone)
}
# what each setting does, for agents (handmouse_settings returns it with value, default and range)
FIELD_HELP = {
    "camera": "Webcam index from handmouse_cameras. Changing it restarts tracking.",
    "smoothing": "Cursor steadiness at rest. Higher = steadier, slightly slower to start moving.",
    "margin": "Share of the camera frame ignored at each edge. Higher = smaller hand moves cross the screen.",
    "pinch": "Thumb-to-finger distance (in hand sizes) that counts as a pinch. Higher = easier, looser.",
    "scroll_speed": "Wheel notches per scroll step.",
    "overlay": "Colored ring around the cursor.",
    "palm_hold": "Seconds of open palm that pause / resume.",
    "drag_hold": "Seconds a pinch is held before it becomes a drag. A quicker pinch is a click.",
    "two_hands": "With both hands in view, one moves the cursor and the other clicks.",
    "pointer_hand": "right or left: the hand that moves the cursor / the game camera.",
    "game_mode": "Stick hand presses direction keys, look hand turns the camera. Ctrl+Alt+G toggles it.",
    "look_speed": "Game camera speed.",
    "stick_dead": "Game stick deadzone in hand sizes. Lower = walks on smaller moves.",
    "predict": "How far the cursor leads a moving hand to hide camera delay. 0 = none; lower it if the cursor overshoots.",
    "fast_camera": "Ask a real webcam for 60 fps (takes effect at start; virtual cameras are left alone).",
    "builtins": "Built-in gestures on/off: move, click, drag, right, scroll, palm, pinch2 (both hands pinched = two-finger touch).",
    "game_keys": "Game bindings per slot (up, down, left, right, pinch, rpinch = stick hand; look_pinch, look_rpinch = look hand): a key ('e'), a combo ('ctrl+r'), lmb / rmb / mmb, or '' to unbind.",
}
PROFILE_KEYS = ("game_keys", "look_speed", "stick_dead", "pinch", "smoothing", "predict", "pointer_hand")
LIMITS = {"camera": (0, 9), "smoothing": (0.0, 0.95), "margin": (0.0, 0.35),
          "pinch": (0.1, 0.6), "scroll_speed": (1, 10), "palm_hold": (0.3, 3.0), "drag_hold": (0.15, 1.5),
          "look_speed": (1.0, 10.0), "stick_dead": (0.15, 0.8), "predict": (0.0, 1.0)}
BUILTINS = ["move", "click", "drag", "right", "scroll", "palm", "pinch2"]
# game bindings: slot -> key, combo, 'lmb' / 'rmb' / 'mmb', or '' (unbound)
GAME_KEYS = {"up": "w", "down": "s", "left": "a", "right": "d", "pinch": "space", "rpinch": "shift",
             "look_pinch": "lmb", "look_rpinch": "rmb"}
MOUSE_BUTTONS = ("lmb", "rmb", "mmb")
ACTIONS = ["none", "click", "double_click", "right_click", "middle_click", "scroll", "keys", "pause"]
PACKAGES = ["mediapipe==0.10.14", "numpy<2", "opencv-contrib-python==4.10.0.84", "pygrabber==0.2"]
# names in DirectShow order = the index OpenCV's CAP_DSHOW opens
LIST_CAMERAS = ("import json\nfrom pygrabber.dshow_graph import FilterGraph\n"
                "print(json.dumps(FilterGraph().get_input_devices()))")
VIRTUAL_CAM = re.compile(r"virtual|obs|meta quest|oculus|snap camera|broadcast|xsplit|mmhmm|droidcam|iriun", re.I)
PY_VERSIONS = ["3.12", "3.11", "3.10", "3.9"]  # mediapipe 0.10.14 has wheels for these
NO_WINDOW = 0x08000000
SETTINGS_KEY = "settings.main"
SETUP_STEPS = [
    ("python", "Find Python 3.9-3.12"),
    ("venv", "Create the helper's Python environment"),
    ("download", "Download mediapipe, opencv and numpy (~100 MB)"),
    ("install", "Install packages"),
    ("verify", "Load the hand tracker"),
    ("selftest", "Run the gesture self-test"),
]
STALE_S = 90  # a "working" setup with no heartbeat for this long was interrupted

# parse_keys / selftest come from the helper source (pure code; cv2 / mediapipe load only in main())
_HELPER_NS: dict = {"__name__": "handmouse_helper"}
exec(compile(helper_src.HELPER, "handmouse_helper", "exec"), _HELPER_NS)
parse_keys = _HELPER_NS["parse_keys"]


# ponytail: the helper drives the cursor with Win32 SendInput; a Quartz CGEvent port is the macOS upgrade
WINDOWS_ONLY = ("Hand Mouse runs on Windows only (it moves the cursor with Win32 input). "
                "On a Mac, run it on the paired Windows PC.")


def _pid_alive(pid: int) -> bool:
    # ponytail: PID reuse could fool this after a reboot-length gap; the stop file still ends ours
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(k.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    finally:
        k.CloseHandle(h)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", str(text or "").strip().lower()).strip("-")[:40]


def register(api) -> None:
    data = api.data
    pid = api.plugin_id
    lock = threading.Lock()

    def base() -> Path:
        p = Path(api.local_dir())
        p.mkdir(parents=True, exist_ok=True)
        return p

    def F(name: str) -> Path:
        return base() / name

    def read_json(name: str, default=None):
        try:
            return json.loads(F(name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def write_json(name: str, value) -> None:
        tmp = F(name + ".tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        os.replace(tmp, F(name))

    def child_env() -> dict:
        # Ducky's own Tcl/Python paths break the venv's tkinter (Tcl version conflict) and imports.
        # Every child also runs with cwd=base(): `python -c` puts the cwd on sys.path, and Ducky's
        # cwd holds its own Python 3.13 extension modules (_ctypes) that crash the venv's Python.
        env = {k: v for k, v in os.environ.items()
               if k.upper() not in ("TCL_LIBRARY", "TK_LIBRARY", "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP")}
        env["PYTHONIOENCODING"] = "utf-8"
        return env

    def venv_python(windowless: bool = False) -> Path:
        if sys.platform != "win32":
            return F("venv") / "bin" / "python"
        return F("venv") / "Scripts" / ("pythonw.exe" if windowless else "python.exe")

    def is_setup() -> bool:
        return (F("venv") / "ready").exists() and venv_python().exists()

    def helper_pid() -> int:
        try:
            return int(F("helper.pid").read_text().strip() or 0)
        except (OSError, ValueError):
            return 0

    def running() -> bool:
        return _pid_alive(helper_pid())

    def log_tail(name: str, n: int = 8) -> list:
        p = F(name)
        if not p.exists():
            return []
        lines = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        return [ln[-300:] for ln in lines[-n:]]

    # ---------- settings ----------

    def clean(fields: dict | None) -> dict:
        out = dict(DEFAULTS, builtins={k: True for k in BUILTINS}, game_keys=dict(GAME_KEYS))
        for k, v in (fields or {}).items():
            if k == "builtins" and isinstance(v, dict):
                out["builtins"].update({b: bool(v[b]) for b in BUILTINS if b in v})
            elif k == "game_keys" and isinstance(v, dict):
                for slot in GAME_KEYS:
                    key = str(v.get(slot, out["game_keys"][slot]) or "").strip().lower()
                    if key and key not in MOUSE_BUTTONS:
                        try:
                            parse_keys(key)
                        except ValueError:
                            continue  # keep the old binding rather than a key the helper can't press
                    out["game_keys"][slot] = key
            elif k in ("overlay", "two_hands", "game_mode", "fast_camera"):
                out[k] = bool(v)
            elif k == "pointer_hand":
                out[k] = "left" if str(v).lower() == "left" else "right"
            elif k in LIMITS:
                try:
                    num = float(v)
                except (TypeError, ValueError):
                    continue
                lo, hi = LIMITS[k]
                num = min(hi, max(lo, num))
                out[k] = int(round(num)) if isinstance(DEFAULTS[k], int) else round(num, 3)
        return out

    def settings() -> dict:
        return clean((data.get(SETTINGS_KEY) or {}).get("values"))

    def gesture_docs() -> list:
        rows = [v for v in data.items("gesture.").values() if isinstance(v, dict)]
        return sorted(rows, key=lambda g: g.get("created", 0))

    def list_cameras(refresh: bool = False) -> dict:
        cached = read_json("cameras.json")
        if cached and not refresh:
            return {"ok": True, "cameras": cached, "selected": settings()["camera"]}
        if not is_setup():
            return {"ok": False, "error": "Run setup first.", "cameras": []}
        try:
            r = subprocess.run([str(venv_python()), "-c", LIST_CAMERAS], capture_output=True, text=True,
                               timeout=30, creationflags=NO_WINDOW, env=child_env(), cwd=str(base()))
            if r.returncode != 0 and "pygrabber" in (r.stderr or ""):
                subprocess.run([str(venv_python()), "-m", "pip", "install", "-q", "--disable-pip-version-check",
                                "pygrabber==0.2"], capture_output=True, timeout=120, creationflags=NO_WINDOW, env=child_env(), cwd=str(base()))
                r = subprocess.run([str(venv_python()), "-c", LIST_CAMERAS], capture_output=True, text=True,
                                   timeout=30, creationflags=NO_WINDOW, env=child_env(), cwd=str(base()))
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-2:]
                return {"ok": False, "error": "Couldn't list cameras: " + " ".join(tail)[-300:], "cameras": []}
            names = json.loads((r.stdout or "[]").strip().splitlines()[-1] or "[]")
        except (subprocess.TimeoutExpired, ValueError, IndexError, OSError) as exc:
            return {"ok": False, "error": f"Couldn't list cameras: {exc}", "cameras": []}
        cams = [{"index": i, "name": str(n), "virtual": bool(VIRTUAL_CAM.search(str(n))),
                 "phone": "Windows Virtual Camera" in str(n)} for i, n in enumerate(names)]
        write_json("cameras.json", cams)
        out = {"ok": True, "cameras": cams, "selected": settings()["camera"]}
        if not cams:
            out["note"] = "Windows reports no cameras."
        elif all(c["virtual"] for c in cams):
            out["note"] = "No built-in or USB webcam found: only virtual cameras (phone, OBS, headsets). They show a picture only while their app is connected."
        return out

    def write_config() -> None:
        cfg = settings()
        cams = read_json("cameras.json") or []
        cam = next((c for c in cams if c.get("index") == cfg["camera"]), None)
        cfg["camera_name"] = cam["name"] if cam else f"Camera {cfg['camera']}"
        cfg["camera_virtual"] = bool(cam["virtual"]) if cam else True  # unknown camera: don't change its mode
        cfg["gestures"] = [{k: g.get(k) for k in ("id", "name", "samples", "tolerance", "hold", "action", "enabled")}
                           for g in gesture_docs()]
        write_json("config.json", cfg)  # a running helper reloads it within half a second

    # ---------- setup with progress ----------

    def setup_state() -> dict:
        st = read_json("setup.json") or {}
        if is_setup() and st.get("phase") != "working":
            st = {**st, "phase": "ready"}
        if st.get("phase") == "working" and time.time() - float(st.get("updated", 0)) > STALE_S:
            st = {**st, "phase": "error", "error": "Setup was interrupted (Ducky closed or the plugin reloaded). Press Set up again; finished downloads are reused."}
        if not st.get("phase"):
            st = {"phase": "idle"}
        st.setdefault("steps", [{"id": i, "label": l, "status": "pending", "detail": ""} for i, l in SETUP_STEPS])
        st.setdefault("percent", 0)
        return st

    class Progress:
        def __init__(self) -> None:
            self.st = {"phase": "working", "percent": 0, "detail": "", "error": "", "started": time.time(),
                       "steps": [{"id": i, "label": l, "status": "pending", "detail": ""} for i, l in SETUP_STEPS],
                       "log": []}
            self.save()

        def save(self) -> None:
            self.st["updated"] = time.time()
            write_json("setup.json", self.st)

        def step(self, sid: str, status: str, detail: str = "", percent=None) -> None:
            for s in self.st["steps"]:
                if s["id"] == sid:
                    s["status"] = status
                    if detail:
                        s["detail"] = detail[:160]
            if detail:
                self.st["detail"] = detail[:160]
            if percent is not None:
                self.st["percent"] = int(percent)
            self.save()
            try:
                from frontend.ui_web.agent_modes import push_ui_event
                push_ui_event({"type": "background_job", "id": f"plugin:{pid}:setup", "source": "plugin",
                               "title": "Hand Mouse setup", "detail": self.st["detail"],
                               "percent": self.st["percent"], "phase": "working", "cancelable": False})
            except Exception:  # the activity row is optional
                pass

        def log(self, line: str) -> None:
            self.st["log"] = (self.st["log"] + [line[-300:]])[-12:]

        def finish(self, ok: bool, error: str = "") -> None:
            self.st["phase"] = "ready" if ok else "error"
            self.st["error"] = error
            if ok:
                self.st["percent"] = 100
                self.st["detail"] = "Ready"
            for s in self.st["steps"]:
                if not ok and s["status"] == "active":
                    s["status"] = "error"
            self.save()
            try:
                from frontend.ui_web.agent_modes import push_ui_event
                push_ui_event({"type": "background_job", "id": f"plugin:{pid}:setup", "source": "plugin",
                               "title": "Hand Mouse setup", "detail": "Ready: press Start" if ok else error[:120],
                               "percent": 100 if ok else None, "phase": "done" if ok else "error", "cancelable": False})
                from frontend.notifications import alert
                alert("Hand Mouse", "Ready: press Start." if ok else "Setup failed.", level="ok" if ok else "error")
            except Exception:
                pass

    def _run_logged(prog: Progress, args: list, on_line=None) -> None:
        with open(F("setup.log"), "a", encoding="utf-8") as fh:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", creationflags=NO_WINDOW, env=child_env(), cwd=str(base()))
            for line in proc.stdout:
                line = line.rstrip()
                fh.write(line + "\n")
                if line:
                    prog.log(line)
                    if on_line:
                        on_line(line)
                    prog.save()
            if proc.wait() != 0:
                raise subprocess.CalledProcessError(proc.returncode, args[:3])

    def _setup_worker() -> None:
        prog = Progress()
        stop_beat = threading.Event()

        def beat() -> None:  # keeps 'updated' fresh during long silent downloads
            while not stop_beat.wait(5):
                prog.save()

        threading.Thread(target=beat, daemon=True, name="handmouse-setup-beat").start()
        F("setup.log").write_text("", encoding="utf-8")
        try:
            prog.step("python", "active", "Looking for the py launcher", 2)
            launcher = shutil.which("py")
            if not launcher:
                raise RuntimeError("Python launcher 'py' not found. Install Python 3.11 from python.org (tick 'py launcher').")
            version = ""
            for v in PY_VERSIONS:
                r = subprocess.run([launcher, f"-{v}", "-c", "import tkinter"], capture_output=True,
                                   creationflags=NO_WINDOW, env=child_env(), cwd=str(base()))
                if r.returncode == 0:
                    version = v
                    break
            if not version:
                raise RuntimeError("Needs Python 3.9-3.12 with tkinter (mediapipe has no wheels for newer). "
                                   "Install Python 3.11 from python.org.")
            prog.step("python", "done", f"Python {version}", 5)

            if venv_python().exists():
                prog.step("venv", "done", "Reusing the existing environment", 10)
            else:
                prog.step("venv", "active", f"python -{version} -m venv", 6)
                shutil.rmtree(F("venv"), ignore_errors=True)
                _run_logged(prog, [launcher, f"-{version}", "-m", "venv", str(F("venv"))])
                prog.step("venv", "done", "Created", 10)

            seen = {"n": 0}
            expected = 28  # ponytail: rough package count for the percent bar; detail shows the real line

            def on_pip(line: str) -> None:
                if line.startswith(("Collecting", "Requirement already satisfied")):
                    seen["n"] += 1
                    name = line.split()[1] if line.startswith("Collecting") else line.split(": ")[1].split()[0]
                    prog.step("download", "active", f"{name} ({seen['n']} of ~{expected})",
                              10 + min(55, 55 * seen["n"] / expected))
                elif line.lstrip().startswith("Downloading"):
                    prog.step("download", "active", line.strip()[:160])
                elif line.startswith("Installing collected packages"):
                    prog.step("download", "done", f"{seen['n']} packages", 65)
                    prog.step("install", "active", line[:160], 70)
                elif line.startswith("Successfully installed"):
                    prog.step("install", "done", line[:160], 88)

            prog.step("download", "active", "Starting pip", 10)
            _run_logged(prog, [str(venv_python()), "-m", "pip", "install", "--disable-pip-version-check",
                               "--progress-bar", "off", *PACKAGES], on_pip)
            for sid in ("download", "install"):
                if next(s for s in prog.st["steps"] if s["id"] == sid)["status"] != "done":
                    prog.step(sid, "done", "Already installed", 88)

            prog.step("verify", "active", "import mediapipe; create the hand tracker", 90)
            r = subprocess.run([str(venv_python()), "-c",
                                "import mediapipe as mp; mp.solutions.hands.Hands(model_complexity=0); print('ok')"],
                               capture_output=True, text=True, creationflags=NO_WINDOW, env=child_env(), cwd=str(base()))
            if r.returncode != 0:
                tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
                raise RuntimeError("The hand tracker didn't load:\n" + "\n".join(tail))
            prog.step("verify", "done", "mediapipe loads", 95)

            prog.step("selftest", "active", "Checking gesture logic", 96)
            F("helper.py").write_text(helper_src.HELPER, encoding="utf-8")
            r = subprocess.run([str(venv_python()), str(F("helper.py")), "--selftest"], capture_output=True,
                               text=True, creationflags=NO_WINDOW, env=child_env(), cwd=str(base()))
            if r.returncode != 0:
                raise RuntimeError("Gesture self-test failed: " + (r.stderr or r.stdout)[-400:])
            prog.step("selftest", "done", "All gestures pass", 99)
            (F("venv") / "ready").write_text(version, encoding="utf-8")
            prog.finish(True)
        except subprocess.CalledProcessError:
            prog.finish(False, "A command failed. The last lines of the log are below.")
        except Exception as exc:
            prog.finish(False, str(exc))
        finally:
            stop_beat.set()

    def setup() -> dict:
        if sys.platform != "win32":
            return {"ok": False, "error": WINDOWS_ONLY}
        with lock:
            if setup_state()["phase"] == "working":
                return {"ok": True, "setup": "working", "note": "Already installing."}
            if running():
                return {"ok": False, "error": "Stop Hand Mouse before reinstalling."}
            try:
                (F("venv") / "ready").unlink()
            except FileNotFoundError:
                pass
            write_json("setup.json", {"phase": "working", "updated": time.time(), "percent": 0})
        threading.Thread(target=_setup_worker, daemon=True, name="handmouse-setup").start()
        return {"ok": True, "setup": "working", "note": "Installing in the background; progress is in handmouse_status."}

    # ---------- run ----------

    def status() -> dict:
        run = running()
        info = read_json("status.json") or {}
        flip = read_json("game_set.json")  # Ctrl+Alt+G in the helper: save it so the panel agrees
        if flip is not None:
            try:
                F("game_set.json").unlink()
                if bool(flip.get("game")) != settings()["game_mode"]:
                    save_settings({"game_mode": bool(flip.get("game"))})
            except FileNotFoundError:
                pass  # the other backend copy took it
        st = setup_state()
        out = {"ok": True, "running": run, "setup": st["phase"],
               "setup_progress": {k: st.get(k) for k in ("percent", "detail", "steps", "error", "log")},
               "mode": (info.get("mode") or "starting") if run else "stopped",
               "fps": info.get("fps", 0) if run else 0, "hand": bool(info.get("hand")) and run,
               "hands": int(info.get("hands") or 0) if run else 0, "two": bool(info.get("two")) and run,
               "paused": bool(info.get("paused")) and run,
               "game": bool(info.get("game")) if run else settings()["game_mode"],
               "held": info.get("held") or [] if run else [],
               # ms: frame arrival -> input sent; out_hz: cursor updates a second (camera fps is `fps`)
               "ms": info.get("ms", 0) if run else 0, "out_hz": info.get("out_hz", 0) if run else 0,
               "match": info.get("match") if run else None, "record": info.get("record") if run else None,
               "events": info.get("events") or [], "last_record": read_json("record_last.json"),
               "settings": settings()}
        if run and info.get("warning"):
            out["warning"] = info["warning"]
        if info.get("mode") == "error":
            out["error"] = info.get("error", "")
        elif not run and helper_pid() and "Traceback" in "\n".join(log_tail("helper.log", 60)):
            lines = [ln for ln in log_tail("helper.log", 60) if not ln.startswith(("W0000", "INFO:", "WARNING: All log"))]
            out["error"] = "The helper stopped:\n" + "\n".join(lines[-6:])
        consume_record()
        return out

    def start() -> dict:
        if sys.platform != "win32":
            return {"ok": False, "error": WINDOWS_ONLY}
        with lock:
            if running():
                return {"ok": True, "running": True, "note": "Already running."}
            if not is_setup():
                return {"ok": False, "error": "Run setup first (Hand Mouse tab → Set up, or handmouse_setup)."}
            F("helper.py").write_text(helper_src.HELPER, encoding="utf-8")
            if not F("cameras.json").exists():
                list_cameras()
            write_config()
            for name in ("stop", "status.json", "record.json"):
                try:
                    F(name).unlink()
                except FileNotFoundError:
                    pass
            with open(F("helper.log"), "w", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    [str(venv_python(windowless=True)), str(F("helper.py")), "--dir", str(base()),
                     "--parent", str(os.getpid())],
                    stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW, env=child_env(), cwd=str(base()))
            F("helper.pid").write_text(str(proc.pid))
        return {"ok": True, "running": True}

    def stop() -> dict:
        hp = helper_pid()
        if not _pid_alive(hp):
            return {"ok": True, "running": False}
        F("stop").write_text("1", encoding="utf-8")
        deadline = time.time() + 3
        while time.time() < deadline and _pid_alive(hp):
            time.sleep(0.1)
        if _pid_alive(hp):
            os.kill(hp, 9)  # TerminateProcess on Windows
        return {"ok": True, "running": False}

    def reset_phone_camera() -> dict:
        """Phone Link's connected camera can hang in 'Connecting to <phone>' for every app (Chrome too).
        Restarting its two COM servers and reopening Phone Link brings the stream back."""
        was = running()
        if was:
            stop()

        def proc_up(name: str) -> bool:
            r = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH"], capture_output=True,
                               text=True, creationflags=NO_WINDOW)
            return name.lower() in (r.stdout or "").lower()

        # Phone Link streams to one app at a time: a browser camera tab (Chrome / Edge keep a
        # video-capture service process) grabs the stream first. End only that service; tabs stay open.
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe' or Name='msedge.exe'\" | "
                        "Where-Object CommandLine -match 'VideoCaptureService' | "
                        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                       capture_output=True, timeout=20, creationflags=NO_WINDOW)
        for name in ("CrossDeviceService.exe", "PhoneExperienceHost.exe"):
            subprocess.run(["taskkill", "/F", "/IM", name], capture_output=True, creationflags=NO_WINDOW)
        time.sleep(1.5)
        try:
            os.startfile(r"shell:AppsFolder\Microsoft.YourPhone_8wekyb3d8bbwe!App")
        except OSError as exc:
            return {"ok": False, "error": f"Couldn't reopen Phone Link: {exc}. Open it from Start, then Start Hand Mouse."}
        deadline = time.time() + 20
        while time.time() < deadline and not proc_up("CrossDeviceService.exe"):
            time.sleep(0.5)
        if not proc_up("CrossDeviceService.exe"):
            return {"ok": False, "error": "Phone Link opened but its camera service didn't start. Check Settings → Bluetooth & devices → Mobile devices → 'Use as connected camera'."}
        time.sleep(2)
        restarted = bool(start().get("ok")) if was else False
        return {"ok": True, "restarted": restarted,
                "note": "Phone camera service restarted. Keep the phone unlocked; the picture comes back within a few seconds."}

    def preview() -> dict:
        p = F("preview.jpg")
        try:
            if not running() or not p.exists() or time.time() - p.stat().st_mtime > 3:
                return {"ok": True, "image": ""}
            return {"ok": True, "image": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()}
        except OSError:
            return {"ok": True, "image": ""}

    def save_settings(fields: dict | None = None) -> dict:
        if not isinstance(fields, dict) or not fields:
            return {"ok": False, "error": "Give fields, e.g. {\"smoothing\": 0.7}. Keys: " + ", ".join(FIELD_HELP)}
        before = data.get(SETTINGS_KEY)
        cur = settings()
        if isinstance(fields.get("builtins"), dict):
            fields = {**fields, "builtins": {**cur["builtins"], **fields["builtins"]}}
        if isinstance(fields.get("game_keys"), dict):
            fields = {**fields, "game_keys": {**cur["game_keys"], **fields["game_keys"]}}
        values = clean({**cur, **fields})
        try:
            data.put(SETTINGS_KEY, {"name": "Hand Mouse settings", "values": values})
        except PermissionError as exc:
            return {"ok": False, "error": str(exc)}
        write_config()
        restarted = False
        if (values["camera"], values["fast_camera"]) != (cur["camera"], cur["fast_camera"]) and running():
            stop()  # the camera opens only at start
            restarted = bool(start().get("ok"))
        api.changeset.record(command="handmouse_save_settings", kind="settings", ident="main", facet="values",
                             before=before, slot=f"{pid}://settings/main/values",
                             inverse=[{"command": "handmouse_save_settings",
                                       "params": {"fields": (before or {}).get("values") or clean({})}}])
        return {"ok": True, "settings": values, "restarted": restarted}

    def describe_settings() -> dict:
        cur, dflt = settings(), clean({})
        fields = {}
        for k, text in FIELD_HELP.items():
            fields[k] = {"value": cur[k], "default": dflt[k], "help": text}
            if k in LIMITS:
                fields[k]["min"], fields[k]["max"] = LIMITS[k]
        return {"ok": True, "settings": cur, "defaults": dflt, "fields": fields}

    # ---------- game profiles: a named set of bindings and feel, one per game ----------

    def profile_rows() -> list:
        rows = [{"id": k.split(".", 1)[1], **v} for k, v in data.items("profile.").items() if isinstance(v, dict)]
        return sorted(rows, key=lambda r: str(r.get("name", "")).lower())

    def profile(action: str = "list", name: str = "", fields: dict | None = None) -> dict:
        action, slug = str(action or "list").lower(), _slug(name)
        if action == "list":
            return {"ok": True, "profiles": profile_rows(), "saved_keys": list(PROFILE_KEYS)}
        if not slug:
            return {"ok": False, "error": "Give the profile a name, e.g. 'Minecraft'."}
        key = f"profile.{slug}"
        before = data.get(key)
        if action == "save":
            cur = settings()
            merged = {**cur, **(fields or {})}
            if isinstance((fields or {}).get("game_keys"), dict):
                merged["game_keys"] = {**cur["game_keys"], **fields["game_keys"]}
            values = {k: v for k, v in clean(merged).items() if k in PROFILE_KEYS}
            doc = {"name": str(name).strip()[:40], "values": values}
            try:
                data.put(key, doc)
            except PermissionError as exc:
                return {"ok": False, "error": str(exc)}
            api.changeset.record(command="handmouse_profile", kind="profile", ident=slug, facet="record",
                                 before=before, slot=f"{pid}://profile/{slug}/record",
                                 inverse=[{"command": "handmouse_profile", "params": (
                                     {"action": "save", "name": before.get("name", slug), "fields": before.get("values")}
                                     if before else {"action": "delete", "name": slug})}])
            return {"ok": True, "profile": {"id": slug, **doc}, "profiles": profile_rows()}
        if before is None:
            return {"ok": False, "error": f"No profile {name!r}. Saved: " + (", ".join(r["name"] for r in profile_rows()) or "none")}
        if action == "apply":
            out = save_settings(dict(before.get("values") or {}))
            return {**out, "applied": before.get("name", slug)}
        if action == "delete":
            try:
                data.delete(key)
            except PermissionError as exc:
                return {"ok": False, "error": str(exc)}
            api.changeset.record(command="handmouse_profile", kind="profile", ident=slug, facet="record",
                                 before=before, slot=f"{pid}://profile/{slug}/record",
                                 inverse=[{"command": "handmouse_profile", "params": {
                                     "action": "save", "name": before.get("name", slug), "fields": before.get("values")}}])
            return {"ok": True, "id": slug, "profiles": profile_rows()}
        return {"ok": False, "error": "action is list, save, apply or delete."}

    # ---------- custom gestures ----------

    def summary(g: dict) -> dict:
        samples = g.get("samples") or []
        return {k: g.get(k) for k in ("id", "name", "tolerance", "hold", "action", "enabled", "created")} | {
            "samples": len(samples), "hands": max(1, len(samples[0]) // 40) if samples else 1,
            "icon": samples[len(samples) // 2] if samples else None}  # one pose fingerprint: the panel draws it

    def clean_action(a) -> dict | str:
        a = a if isinstance(a, dict) else {"type": str(a or "none")}
        kind = str(a.get("type") or "none")
        if kind not in ACTIONS:
            return f"Unknown action {kind!r}. Use one of: {', '.join(ACTIONS)}."
        out = {"type": kind}
        if kind == "keys":
            try:
                parse_keys(a.get("keys", ""))
            except ValueError as exc:
                return f"Keys: {exc}. Example: ctrl+tab, alt+f4, enter, ctrl+shift+t."
            out["keys"] = str(a["keys"]).strip().lower()
            if a.get("hold"):
                out["hold"] = True  # keys stay down while the pose lasts (games), not one tap
        if kind == "scroll":
            try:
                out["amount"] = max(-20, min(20, int(a.get("amount", 3)))) or 3
            except (TypeError, ValueError):
                out["amount"] = 3
        return out

    def list_gestures() -> dict:
        return {"ok": True, "gestures": [summary(g) for g in gesture_docs()], "actions": ACTIONS,
                "builtins": settings()["builtins"]}

    def save_gesture(gesture_id: str = "", fields: dict | None = None) -> dict:
        gid = _slug(gesture_id)
        before = data.get(f"gesture.{gid}") if gid else None
        if before is None:
            return {"ok": False, "error": f"No gesture {gesture_id!r}. Record one with handmouse_record_gesture."}
        fields = fields or {}
        doc = dict(before)
        if "name" in fields and str(fields["name"]).strip():
            doc["name"] = str(fields["name"]).strip()[:60]
        if "action" in fields:
            act = clean_action(fields["action"])
            if isinstance(act, str):
                return {"ok": False, "error": act}
            doc["action"] = act
        for k, lo, hi in (("tolerance", 0.03, 0.5), ("hold", 0.1, 3.0)):
            if k in fields:
                try:
                    doc[k] = round(min(hi, max(lo, float(fields[k]))), 3)
                except (TypeError, ValueError):
                    return {"ok": False, "error": f"{k} must be a number."}
        if "enabled" in fields:
            doc["enabled"] = bool(fields["enabled"])
        try:
            data.put(f"gesture.{gid}", doc)
        except PermissionError as exc:
            return {"ok": False, "error": str(exc)}
        write_config()
        api.changeset.record(command="handmouse_save_gesture", kind="gesture", ident=gid, facet="record",
                             before=summary(before), slot=f"{pid}://gesture/{gid}/record",
                             inverse=[{"command": "handmouse_save_gesture",
                                       "params": {"gesture_id": gid, "fields": {k: before.get(k) for k in
                                                  ("name", "action", "tolerance", "hold", "enabled")}}}])
        return {"ok": True, "gesture": summary(doc)}

    def delete_gesture(gesture_id: str = "") -> dict:
        gid = _slug(gesture_id)
        before = data.get(f"gesture.{gid}") if gid else None
        if before is None:
            return {"ok": False, "error": f"No gesture {gesture_id!r}."}
        try:
            data.delete(f"gesture.{gid}")
        except PermissionError as exc:
            return {"ok": False, "error": str(exc)}
        write_config()
        # ponytail: undo restores settings, not samples (re-record); samples are too big for the ledger
        api.changeset.record(command="handmouse_delete_gesture", kind="gesture", ident=gid, facet="record",
                             before=summary(before), slot=f"{pid}://gesture/{gid}/record", inverse=[])
        return {"ok": True, "id": gid}

    def record_gesture(name: str = "", gesture_id: str = "", seconds: float = 2.0, action: dict | None = None) -> dict:
        existing = data.get(f"gesture.{_slug(gesture_id)}") if gesture_id else None
        if gesture_id and existing is None:
            return {"ok": False, "error": f"No gesture {gesture_id!r} to re-record."}
        name = str(name or (existing or {}).get("name") or "").strip()[:60]
        if not name:
            return {"ok": False, "error": "Give the gesture a name, e.g. 'Peace sign'."}
        act = clean_action(action or (existing or {}).get("action") or {"type": "none"})
        if isinstance(act, str):
            return {"ok": False, "error": act}
        gid = existing["id"] if existing else f"{_slug(name) or 'gesture'}-{int(time.time()) % 100000}"
        if not running():
            out = start()
            if not out.get("ok"):
                return out
        try:
            secs = max(1.0, min(5.0, float(seconds or 2)))
        except (TypeError, ValueError):
            secs = 2.0
        try:
            F("record_cancel").unlink()  # a stale cancel must not stop this recording
        except FileNotFoundError:
            pass
        write_json("record_pending.json", {"id": gid, "name": name, "action": act})
        write_json("record.json", {"id": gid, "name": name, "seconds": secs, "countdown": 3})
        return {"ok": True, "id": gid, "note": f"Hold the pose in view: 3 s countdown, then {secs:g} s of recording."}

    def cancel_record() -> dict:
        try:
            F("record.json").unlink()
        except FileNotFoundError:
            pass
        F("record_cancel").write_text("1", encoding="utf-8")
        return {"ok": True}

    def consume_record() -> None:
        res = read_json("record_result.json")
        if not res:
            return
        try:
            F("record_result.json").unlink()
        except FileNotFoundError:
            return  # the other backend copy took it
        pending = read_json("record_pending.json") or {}
        gid = res.get("id")
        if res.get("error") or not res.get("samples"):
            write_json("record_last.json", {"id": gid, "ok": False, "error": res.get("error", "No samples."), "t": time.time()})
            return
        key = f"gesture.{gid}"
        before = data.get(key)
        doc = {**(before or {"tolerance": 0.15, "hold": 0.4, "enabled": True, "created": time.time()}),
               "id": gid, "name": res.get("name") or pending.get("name") or gid,
               "samples": res["samples"], "action": (before or {}).get("action") or pending.get("action") or {"type": "none"}}
        try:
            data.put(key, doc)
        except PermissionError as exc:
            write_json("record_last.json", {"id": gid, "ok": False, "error": str(exc), "t": time.time()})
            return
        write_config()
        api.changeset.record(command="handmouse_record_gesture", kind="gesture", ident=gid, facet="samples",
                             before=summary(before) if before else None, slot=f"{pid}://gesture/{gid}/samples",
                             inverse=[] if before else [{"command": "handmouse_delete_gesture", "params": {"gesture_id": gid}}])
        write_json("record_last.json", {"id": gid, "ok": True, "samples": len(res["samples"]), "t": time.time()})

    # ---------- MCP tools ----------

    INTENT = r"\b(hand ?mouse|hand tracking|finger tracking|gestures?)\b"

    @api.tool(intent=INTENT, listener=False)
    def handmouse_status() -> dict:
        """Hand Mouse state: setup progress, running, gesture mode, matched custom gesture, recent activity, and speed (fps = camera frames a second, ms = delay from frame to input, out_hz = cursor updates a second)."""
        return status()

    @api.tool(intent=INTENT, listener=False)
    def handmouse_setup() -> dict:
        """Install Hand Mouse's webcam helper once (Python venv with mediapipe + opencv, ~100 MB download)."""
        return setup()

    @api.tool(intent=INTENT, listener=False)
    def handmouse_start() -> dict:
        """Start webcam hand tracking: it takes over the Windows cursor until stopped or palm-paused."""
        return start()

    @api.tool(intent=INTENT, listener=False)
    def handmouse_stop() -> dict:
        """Stop Hand Mouse webcam tracking and give the cursor back."""
        return stop()

    @api.tool(intent=INTENT, listener=False)
    def handmouse_settings() -> dict:
        """Read every Hand Mouse setting with its value, default, range and what it does (fields), ready to change with handmouse_save_settings."""
        return describe_settings()

    @api.tool(intent=INTENT, listener=False)
    def handmouse_save_settings(fields: dict | None = None) -> dict:
        """Change Hand Mouse settings or turn built-in gestures on/off (builtins: move, click, drag, right, scroll, palm, pinch2 = both hands pinched -> two-finger touch to resize/rotate/pan). drag_hold: seconds a pinch must be held to start a drag (quicker = click). game_mode: true = stick hand presses direction keys, look hand moves the camera; game_keys {up, down, left, right, pinch, rpinch, look_pinch, look_rpinch}: a key, combo, lmb/rmb/mmb or '' each; look_speed 1-10; stick_dead 0.15-0.8. predict 0-1: how far the cursor leads a moving hand (lower if it overshoots); smoothing 0-0.95; fast_camera: 60 fps on real webcams. handmouse_settings lists every field with range and meaning. two_hands: true = with both hands in view one moves, the other clicks; pointer_hand: right|left."""
        return save_settings(fields)

    @api.tool(intent=INTENT, listener=False)
    def handmouse_preview() -> dict:
        """Latest Hand Mouse camera frame with the tracked hand drawn on it, as a JPEG data URI."""
        return preview()

    @api.tool(intent=INTENT, listener=False)
    def handmouse_gestures() -> dict:
        """List Hand Mouse custom gestures (name, action, tolerance, hold, enabled) and built-in toggles."""
        return list_gestures()

    @api.tool(intent=INTENT, listener=False)
    def handmouse_record_gesture(name: str = "", gesture_id: str = "", seconds: float = 2.0,
                                 action: dict | None = None) -> dict:
        """Record a new custom hand pose from the webcam (or re-record gesture_id). action: {type: click|double_click|right_click|middle_click|scroll|keys|pause|none, keys: 'ctrl+tab', amount: 3}."""
        return record_gesture(name, gesture_id, seconds, action)

    @api.tool(intent=INTENT, listener=False)
    def handmouse_cancel_record() -> dict:
        """Cancel a Hand Mouse gesture recording that is counting down or capturing."""
        return cancel_record()

    @api.tool(intent=INTENT, listener=False)
    def handmouse_save_gesture(gesture_id: str = "", fields: dict | None = None) -> dict:
        """Edit a Hand Mouse custom gesture: name, action, tolerance (0.03-0.5), hold seconds, enabled."""
        return save_gesture(gesture_id, fields)

    @api.tool(intent=INTENT, listener=False)
    def handmouse_delete_gesture(gesture_id: str = "") -> dict:
        """Delete a Hand Mouse custom gesture."""
        return delete_gesture(gesture_id)

    @api.tool(intent=INTENT, listener=False)
    def handmouse_profile(action: str = "list", name: str = "", fields: dict | None = None) -> dict:
        """Hand Mouse game profiles, one per game (controls preset: game_keys, look_speed, stick_dead, pinch, smoothing, predict, pointer_hand). action: list | save (current settings, plus fields to override, e.g. {game_keys: {pinch: 'e'}}) | apply | delete. name: e.g. 'Minecraft'."""
        return profile(action, name, fields)

    # ---------- workflow nodes ----------

    @api.register_pipeline_node("handmouse.start")
    def node_start(ctx: dict) -> dict:
        cfg = ctx.get("config") or {}
        if not cfg.get("confirm"):
            return {"ok": False, "running": running(), "error": "Turn on 'Take over the cursor' in the node to start tracking."}
        out = start()
        return {**out, "running": bool(out.get("running"))}

    @api.register_pipeline_node("handmouse.stop")
    def node_stop(ctx: dict) -> dict:
        return stop()

    @api.register_pipeline_node("handmouse.status")
    def node_status(ctx: dict) -> dict:
        return status()

    @api.register_pipeline_node("handmouse.gestures")
    def node_gestures(ctx: dict) -> dict:
        out = list_gestures()
        return {**out, "count": len(out["gestures"])}

    @api.tool(intent=INTENT, listener=False)
    def handmouse_running() -> dict:
        """Quick check whether Hand Mouse tracking is running right now (drives the header's red dot)."""
        return {"ok": True, "running": running()}

    @api.tool(intent=INTENT, listener=False)
    def handmouse_cameras(refresh: bool = False) -> dict:
        """List the PC's cameras by name (index = Hand Mouse 'camera' setting), flagging virtual ones (phone, OBS, headsets)."""
        return list_cameras(bool(refresh))

    @api.tool(intent=INTENT, listener=False)
    def handmouse_reset_phone_camera() -> dict:
        """Fix a phone camera (Phone Link) stuck on 'Connecting' / black spinner: restarts Phone Link's camera service and reopens Phone Link."""
        return reset_phone_camera()

    api.register_panel_rpc("reset_phone_camera", lambda: reset_phone_camera())
    api.register_panel_rpc("cameras", lambda refresh=False: list_cameras(bool(refresh)))
    api.register_panel_rpc("running", lambda: {"ok": True, "running": running()})
    api.register_panel_rpc("status", lambda: status())
    api.register_panel_rpc("setup", lambda: setup())
    api.register_panel_rpc("start", lambda: start())
    api.register_panel_rpc("stop", lambda: stop())
    api.register_panel_rpc("settings", lambda: {"ok": True, "settings": settings(), "defaults": clean({})})
    api.register_panel_rpc("profile", lambda action="list", name="", fields=None: profile(action, name, fields))
    api.register_panel_rpc("save_settings", lambda fields=None: save_settings(fields))
    api.register_panel_rpc("preview", lambda: preview())
    api.register_panel_rpc("gestures", lambda: list_gestures())
    api.register_panel_rpc("record_gesture", lambda name="", gesture_id="", seconds=2.0, action=None:
                           record_gesture(name, gesture_id, seconds, action))
    api.register_panel_rpc("cancel_record", lambda: cancel_record())
    api.register_panel_rpc("save_gesture", lambda gesture_id="", fields=None: save_gesture(gesture_id, fields))
    api.register_panel_rpc("delete_gesture", lambda gesture_id="": delete_gesture(gesture_id))
