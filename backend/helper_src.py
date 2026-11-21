"""Source of the webcam helper. The backend writes it to the plugin's local folder and
runs it in its own venv (mediapipe + opencv), so the app's Python never imports them."""

HELPER = r'''
import collections, ctypes, ctypes.wintypes, json, math, os, sys, threading, time

# ---------- geometry ----------

def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def read_hand(p):
    """p: 21 (x, y) MediaPipe landmarks, normalized, image already mirrored."""
    # wrist -> middle knuckle shrinks when the hand tilts toward the camera; palm width doesn't
    size = max(_d(p[0], p[9]), _d(p[5], p[17])) or 1e-6
    return {
        "pinch": _d(p[4], p[8]) / size,      # thumb + index
        "rpinch": _d(p[4], p[12]) / size,    # thumb + middle
        "fingers": [p[t][1] < p[t - 2][1] for t in (8, 12, 16, 20)],
        "point": ((p[4][0] + p[8][0]) / 2, (p[4][1] + p[8][1]) / 2),
        # knuckle row: it doesn't move when the fingers pinch, so a click lands where you aimed,
        # and four landmarks averaged shake half as much as one
        "palm": (sum(p[i][0] for i in (5, 9, 13, 17)) / 4, sum(p[i][1] for i in (5, 9, 13, 17)) / 4),
        "tip": (p[8][0], p[8][1]),           # index fingertip: the pointer in two-hand mode
        "tip_y": p[8][1],
    }


def features(p):
    """Pose fingerprint: landmarks 1-20 relative to the wrist, scaled by hand size and
    rotated so wrist -> middle knuckle points up. Same pose anywhere / any roll = same vector."""
    wx, wy = p[0]
    ax, ay = p[9][0] - wx, p[9][1] - wy
    size = math.hypot(ax, ay) or 1e-6
    # rotate (ax, ay) onto (0, -1)
    c, s = -ay / size, -ax / size
    out = []
    for x, y in p[1:]:
        dx, dy = (x - wx) / size, (y - wy) / size
        out.append(round(dx * c - dy * s, 4))
        out.append(round(dx * s + dy * c, 4))
    return out


def pose_distance(a, b):
    """RMS landmark distance in hand-size units (same pose ~0.05, different ~0.3+)."""
    n = min(len(a), len(b)) // 2 or 1
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(2 * n)) / n)


# ---------- custom gestures ----------

class Matcher:
    """Static poses recorded by the person. A pose held for `hold` seconds fires once."""

    def __init__(self, gestures):
        self.set(gestures)

    def set(self, gestures):
        self.last_idx = getattr(self, "last_idx", None)
        self.gestures = [g for g in (gestures or []) if g.get("enabled", True) and g.get("samples")]
        keep = {g["id"] for g in self.gestures}
        self.since = {k: v for k, v in getattr(self, "since", {}).items() if k in keep}
        self.fired = {k for k in getattr(self, "fired", set()) if k in keep}

    def best(self, feats):
        """feats: candidate fingerprints (both hands together, then each hand). A sample only
        compares with a candidate of its own length, so 1-hand and 2-hand gestures never mix."""
        best = (None, 9.0)
        self.last_idx = None  # which candidate matched: 0 = both hands, 1 = pointer hand, 2 = other hand
        for g in self.gestures:
            for idx, feat in enumerate(feats):
                # ponytail: nearest sample, linear scan; fine for dozens of gestures x 40 samples
                ds = [pose_distance(feat, s) for s in g["samples"] if len(s) == len(feat)]
                if ds and min(ds) < best[1]:
                    best = (g, min(ds))
                    self.last_idx = idx
        return best

    def step(self, feats, now):
        """feats: one fingerprint or a list of candidates.
        -> (matched gesture or None, distance, fire gesture or None)"""
        if feats and isinstance(feats[0], (int, float)):
            feats = [feats]
        if not feats:
            self.since.clear()
            self.fired.clear()
            return None, 9.0, None
        g, d = self.best(feats)
        hit = g if g is not None and d <= float(g.get("tolerance", 0.15)) else None
        for gid in list(self.since):
            if hit is None or gid != hit["id"]:
                self.since.pop(gid, None)
                self.fired.discard(gid)
        if hit is None:
            return None, d, None
        t0 = self.since.setdefault(hit["id"], now)
        if hit["id"] not in self.fired and now - t0 >= float(hit.get("hold", 0.4)):
            self.fired.add(hit["id"])
            return hit, d, hit
        return hit, d, None


# ---------- pointer filter ----------

class OneEuro:
    """One Euro filter: heavy smoothing when the hand is nearly still (no shake),
    almost none when it moves fast (no lag). min_cutoff = steadiness, beta = speed response."""

    def __init__(self, min_cutoff=1.5, beta=8.0, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _a(cutoff, dt):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def reset(self):
        self.x = None

    def __call__(self, x, t):
        if self.x is None or t <= self.t:
            if self.x is None:
                self.x, self.dx = x, 0.0
            self.t = t
            return self.x
        dt = t - self.t
        self.dx += self._a(self.dc, dt) * ((x - self.x) / dt - self.dx)
        self.x += self._a(self.mc + self.beta * abs(self.dx), dt) * (x - self.x)
        self.t = t
        return self.x


class HandSmoother:
    """MediaPipe Hands doesn't smooth its landmarks, so pinch distances and poses flicker.
    One Euro on every landmark, per hand (a hand is followed by its wrist): steady at rest,
    no lag in motion. A hand that drops out for a moment keeps its filters."""

    def __init__(self, min_cutoff=4.0, beta=40.0):
        self.mc, self.beta, self.slots = min_cutoff, beta, []

    def __call__(self, hands, t):
        free = [s for s in self.slots if t - s["t"] < 0.3]
        used, out = [], []
        for p in hands:
            s = min(free, key=lambda s: _d(s["w"], p[0]), default=None)
            if s is not None and _d(s["w"], p[0]) < 0.25:
                free.remove(s)
            else:
                s = {"f": [OneEuro(self.mc, self.beta) for _ in range(42)]}
            s["w"], s["t"] = p[0], t
            out.append([(s["f"][2 * i](x, t), s["f"][2 * i + 1](y, t)) for i, (x, y) in enumerate(p)])
            used.append(s)
        self.slots = used + free
        return out


# ---------- built-in gestures ----------

class Gestures:
    SCROLL_STEP = 0.03  # normalized hand travel per wheel notch

    def __init__(self, cfg):
        self.cfg = cfg
        self.freeze_until = 0.0  # cursor holds still right after a press / release
        self.down = self.rdown = self.paused = False
        self.palm_since = None
        self.scroll_y = None
        self.mode = "nohand"
        self.since = {}  # debounce: when each pinch state started
        self.r_ready_at = 0.0
        self.pinch_at = None  # when the current pinch started: quick release = click, held = drag
        self.touch = False  # both hands pinched: a two-finger touch is in progress
        self.block_click = False  # after a touch ends, the click hand must open before it clicks again
        self.lost_at = None  # when the pointer hand went out of view

    LOST_GRACE = 0.15  # the tracker drops a hand for a frame or two now and then: hold on that long
    DEBOUNCE = 0.08  # a pinch must hold this long (2-3 frames) before it counts: no flicker clicks

    def held(self, key, cond, now):
        """True once `cond` has been true for DEBOUNCE seconds in a row."""
        if not cond:
            self.since.pop(key, None)
            return False
        return now - self.since.setdefault(key, now) >= self.DEBOUNCE

    def on(self, name):
        return bool((self.cfg.get("builtins") or {}).get(name, True))

    def release(self):
        out = []
        if self.down:
            out.append(("up",))
            self.down = False
        if self.touch:
            out.append(("touch_up",))
            self.touch = False
        self.pinch_at = None
        self.scroll_y = None
        return out

    def step(self, p, now, click_p=None, two=None):
        """p: the pointer hand. click_p: the other hand in two-hand mode: it alone clicks,
        so p only moves (pinching never nudges the cursor). two=True with click_p None: the
        click hand dropped out for a moment; keep moving with the fingertip, no clicks."""
        out = []
        if p is None:
            if self.lost_at is None:
                self.lost_at = now
            if (self.down or self.touch) and now - self.lost_at < self.LOST_GRACE:
                return [("touch_hold",)] if self.touch else []  # a dropped frame never drops what you hold
            out = self.release()
            self.palm_since = None
            self.mode = "paused" if self.paused else "nohand"
            return out
        self.lost_at = None
        h = read_hand(p)
        f = h["fingers"]
        two = click_p is not None if two is None else two
        c = read_hand(click_p) if click_p is not None else (None if two else h)  # the hand that clicks
        on = float(self.cfg.get("pinch", 0.3))
        off = on * 1.6  # wide gap between press and release so noise near the line can't toggle it

        # open palm held = pause / resume (once per palm)
        if self.on("palm") and all(f) and h["pinch"] > 0.6 and h["rpinch"] > 0.6 and not self.down and not self.touch:
            if self.palm_since is None:
                self.palm_since = now
            elif now - self.palm_since >= float(self.cfg.get("palm_hold", 1.0)):
                self.paused = not self.paused
                self.palm_since = math.inf
                out.append(("pause",) if self.paused else ("resume",))
        else:
            self.palm_since = None
        if self.paused:
            self.mode = "paused"
            return out

        # both hands pinched = one two-finger touch: pull apart / together to resize or zoom,
        # twist to rotate, move both to pan. It ends when either hand opens.
        if self.touch and (not two or not self.on("pinch2")):
            self.touch = False
            out.append(("touch_up",))
        if two and self.on("pinch2"):
            if self.touch and c is None:  # click hand lost for a moment: keep the fingers where they are
                self.mode = "touch"
                out.append(("touch_hold",))
                return out
            if (c is not None and not self.touch
                    and self.held("t_on", h["pinch"] < on and c["pinch"] < on, now)):
                self.since.pop("t_on", None)
                self.since.pop("down", None)
                if self.down:
                    out.append(("up",))
                self.down, self.pinch_at, self.touch = False, None, True
                self.mode = "touch"
                out.append(("touch_down", h["point"], c["point"]))
                return out
            if self.touch:
                if self.held("t_off", h["pinch"] > off or c["pinch"] > off, now):
                    self.since.pop("t_off", None)
                    self.touch, self.block_click = False, True
                    out.append(("touch_up",))
                else:
                    self.mode = "touch"
                    out.append(("touch_move", h["point"], c["point"]))
                    return out

        # two fingers up (index + middle), ring + pinky folded = scroll
        if (self.on("scroll") and f[0] and f[1] and not f[2] and not f[3]
                and h["pinch"] > off and h["rpinch"] > off and not self.down):
            if self.scroll_y is None:
                self.scroll_y = h["tip_y"]
            steps = int((h["tip_y"] - self.scroll_y) / self.SCROLL_STEP)
            if steps:
                out.append(("scroll", -steps * int(self.cfg.get("scroll_speed", 1))))
                self.scroll_y += steps * self.SCROLL_STEP
            self.rdown = False
            self.mode = "scroll"
            return out
        self.scroll_y = None

        # one hand: the knuckles steer (pinching doesn't move them), and the cursor holds still
        # for a moment after press / release; a held pinch that moves on is a drag
        clicks = self.on("click") or self.on("drag")
        if self.on("move") and (two or now >= self.freeze_until):
            out.append(("move",) + (h["tip"] if two else h["palm"]))
        if c is None:  # click hand briefly lost: hold the button state as it is
            self.mode = "drag" if self.down else "move"
            return out
        # click and drag are separate: a quick pinch clicks on release; a pinch held for
        # drag_hold seconds presses the button (drag) and lets go when the pinch opens
        pressed = self.pinch_at is not None or self.down
        if self.block_click:
            if c["pinch"] > off:
                self.block_click = False
        elif clicks:
            if not pressed and self.held("down", c["pinch"] < on and c["pinch"] <= c["rpinch"], now):
                self.pinch_at = now
                self.since.pop("down", None)
                self.freeze_until = now + 0.2
            elif pressed and self.held("up", c["pinch"] > off, now):
                self.since.pop("up", None)
                if self.down:
                    out.append(("up",))
                elif self.on("click"):
                    out.append(("click",))
                self.down, self.pinch_at = False, None
                self.freeze_until = now + 0.15
            elif (pressed and not self.down and self.on("drag")
                    and now - self.pinch_at >= float(self.cfg.get("drag_hold", 0.35))):
                self.down = True
                out.append(("down",))
            pressed = self.pinch_at is not None or self.down
        if self.on("right") and not pressed:
            if (not self.rdown and now >= self.r_ready_at
                    and self.held("right", c["rpinch"] < on and c["rpinch"] < c["pinch"], now)):
                self.rdown = True
                self.since.pop("right", None)
                self.r_ready_at = now + 0.5  # one right click per pinch, at most 2 a second
                out.append(("right",))
            elif self.rdown and self.held("rup", c["rpinch"] > off, now):
                self.rdown = False
                self.since.pop("rup", None)
        self.mode = "drag" if self.down else "move"
        return out


def assign_hands(hands, ptr_right, last, now, grace=0.6):
    """Sticky two-hand roles. hands: sorted left -> right. last: memory of the previous two-hand
    frame (updated here). -> (pointer, click hand, two). Roles start from screen side, then follow
    each wrist, so a hand that drops out for a frame never swaps which hand moves and which clicks."""
    near = lambda p, w: _d(p[0], w)
    recent = last.get("t") is not None and now - last["t"] < grace
    if len(hands) == 2:
        a, b = hands
        if recent:
            keep = near(b, last["ptr"]) + near(a, last["click"]) <= near(a, last["ptr"]) + near(b, last["click"])
            ptr, clk = (b, a) if keep else (a, b)
        else:
            ptr, clk = (b, a) if ptr_right else (a, b)
        last.update(ptr=ptr[0], click=clk[0], t=now)
        return ptr, clk, True
    if len(hands) == 1 and recent:
        p = hands[0]
        if near(p, last["ptr"]) <= near(p, last["click"]):
            last["ptr"] = p[0]
            return p, None, True  # click hand dropped out for a moment
        return None, None, True  # only the click hand in view: it must not grab the cursor
    return (hands[0] if hands else None), None, False


# ---------- two-finger touch ----------

TOUCH_GAP = 200  # px between the two fingers when the pinch starts, centered on the cursor


def touch_geometry(a, b, anchor, fw, fh):
    """a, b: the two pinch points (normalized, filtered). anchor: the start of the touch
    {cx, cy (cursor px), mx, my (start midpoint, normalized), len0 (start gap, frame px), kx, ky}.
    -> two screen points: centered on the cursor, TOUCH_GAP apart along the hands' line at the
    start, scaled by how far the hands spread, turned with them, panned as their midpoint moves."""
    vx, vy = (b[0] - a[0]) * fw, (b[1] - a[1]) * fh
    n = math.hypot(vx, vy) or 1e-6
    half = TOUCH_GAP / 2 * n / max(anchor["len0"], 1e-3)
    ux, uy = vx / n, vy / n
    cx = anchor["cx"] + ((a[0] + b[0]) / 2 - anchor["mx"]) * anchor["kx"]
    cy = anchor["cy"] + ((a[1] + b[1]) / 2 - anchor["my"]) * anchor["ky"]
    return [(cx - ux * half, cy - uy * half), (cx + ux * half, cy + uy * half)]


class POINTER_INFO(ctypes.Structure):
    _fields_ = [("pointerType", ctypes.c_uint32), ("pointerId", ctypes.c_uint32), ("frameId", ctypes.c_uint32),
                ("pointerFlags", ctypes.c_uint32), ("sourceDevice", ctypes.wintypes.HANDLE),
                ("hwndTarget", ctypes.wintypes.HWND), ("ptPixelLocation", ctypes.wintypes.POINT),
                ("ptHimetricLocation", ctypes.wintypes.POINT), ("ptPixelLocationRaw", ctypes.wintypes.POINT),
                ("ptHimetricLocationRaw", ctypes.wintypes.POINT), ("dwTime", ctypes.wintypes.DWORD),
                ("historyCount", ctypes.c_uint32), ("InputData", ctypes.c_int32),
                ("dwKeyStates", ctypes.wintypes.DWORD), ("PerformanceCount", ctypes.c_uint64),
                ("ButtonChangeType", ctypes.c_int)]


class POINTER_TOUCH_INFO(ctypes.Structure):
    _fields_ = [("pointerInfo", POINTER_INFO), ("touchFlags", ctypes.c_uint32), ("touchMask", ctypes.c_uint32),
                ("rcContact", ctypes.wintypes.RECT), ("rcContactRaw", ctypes.wintypes.RECT),
                ("orientation", ctypes.c_uint32), ("pressure", ctypes.c_uint32)]


class Touch:
    """Windows touch injection: apps see a real two-finger touchscreen gesture."""
    DOWN, UPDATE, UP, INRANGE, INCONTACT = 0x10000, 0x20000, 0x40000, 0x2, 0x4

    def __init__(self, user32):
        self.u = user32
        self.ok = bool(user32.InitializeTouchInjection(2, 1))  # 2 contacts, default visual feedback
        self.last = None

    def _send(self, pts, flags):
        arr = (POINTER_TOUCH_INFO * len(pts))()
        for i, (x, y) in enumerate(pts):
            x, y = int(x), int(y)
            t = arr[i]
            t.pointerInfo.pointerType = 2  # PT_TOUCH
            t.pointerInfo.pointerId = i
            t.pointerInfo.pointerFlags = flags
            t.pointerInfo.ptPixelLocation.x, t.pointerInfo.ptPixelLocation.y = x, y
            t.touchMask = 0x7  # contact area, orientation, pressure
            t.rcContact.left, t.rcContact.top, t.rcContact.right, t.rcContact.bottom = x - 2, y - 2, x + 2, y + 2
            t.orientation, t.pressure = 90, 512
        return bool(self.u.InjectTouchInput(len(pts), arr))

    def down(self, pts):
        self.last = pts
        return self._send(pts, self.DOWN | self.INRANGE | self.INCONTACT)

    def move(self, pts=None):
        if self.last is None:
            return False
        self.last = pts or self.last
        return self._send(self.last, self.UPDATE | self.INRANGE | self.INCONTACT)

    def up(self):
        if self.last is None:
            return False
        pts, self.last = self.last, None
        return self._send(pts, self.UP)


# ---------- game mode ----------

# default bindings: the stick hand (left) moves, the look hand (right) aims. "" = unbound
GAME_KEYS = {"up": "w", "down": "s", "left": "a", "right": "d", "pinch": "space", "rpinch": "shift",
             "look_pinch": "lmb", "look_rpinch": "rmb"}


class Game:
    """Stick hand = a thumbstick (palm offset from where it appeared -> direction keys);
    look hand = relative mouse movement (camera); pinches on either hand = buttons."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.center = self.size = None
        self.dirs = set()
        self.pin = {}
        self.fx, self.fy = OneEuro(1.0, 4.0), OneEuro(1.0, 4.0)
        self.prev = None
        self.rem = [0.0, 0.0]
        self.seen = {}

    GRACE = 0.15  # a hand the tracker drops for a frame or two keeps its keys down that long

    def _lost(self, key, p, now):
        """True while a hand that just vanished still counts as there."""
        if p is not None:
            self.seen[key] = now
            return False
        return now is not None and self.seen.get(key) is not None and now - self.seen[key] < self.GRACE

    def stick(self, p, dead, now=None):
        """-> set of 'up' / 'down' / 'left' / 'right'. dead: deadzone in hand sizes;
        a held direction lets go only at 70% of it, so the edge never stutters."""
        if self._lost("dir", p, now):
            return set(self.dirs)
        if p is None:
            self.center, self.dirs = None, set()
            return set()
        size = max(_d(p[0], p[9]), _d(p[5], p[17])) or 1e-6
        if self.center is None:
            self.center = p[9]
        self.size = size
        dx, dy = (p[9][0] - self.center[0]) / size, (p[9][1] - self.center[1]) / size
        out = set()
        for name, val in (("up", -dy), ("down", dy), ("left", -dx), ("right", dx)):
            if val > (dead * 0.7 if name in self.dirs else dead):
                out.add(name)
        self.dirs = out
        return set(out)

    def pinches(self, side, p, on, off, now=None):
        """-> {'pinch', 'rpinch'} held on this hand, with press / release hysteresis."""
        if self._lost(side, p, now):
            return set(self.pin.get(side, ()))
        if p is None:
            self.pin[side] = set()
            return set()
        h = read_hand(p)
        cur = self.pin.setdefault(side, set())
        for k, other in (("pinch", "rpinch"), ("rpinch", "pinch")):
            if k in cur and h[k] > off:
                cur.discard(k)
            elif k not in cur and h[k] < on and h[k] <= h[other]:
                cur.add(k)
        return set(cur)

    def look(self, p, now, gain):
        """-> (dx, dy) relative mouse pixels from the look hand's motion (gain = px per frame width)."""
        if p is None:
            self.prev = None
            self.fx.reset()
            self.fy.reset()
            return 0, 0
        x, y = self.fx(p[9][0], now), self.fy(p[9][1], now)
        if self.prev is None:
            self.prev = (x, y)
            return 0, 0
        self.rem[0] += (x - self.prev[0]) * gain
        self.rem[1] += (y - self.prev[1]) * gain
        self.prev = (x, y)
        mx, my = int(self.rem[0]), int(self.rem[1])
        self.rem[0] -= mx
        self.rem[1] -= my
        return mx, my


# ---------- keys ----------

VK = {"ctrl": 0x11, "control": 0x11, "shift": 0x10, "alt": 0x12, "win": 0x5B, "tab": 0x09,
      "enter": 0x0D, "return": 0x0D, "esc": 0x1B, "escape": 0x1B, "space": 0x20, "backspace": 0x08,
      "delete": 0x2E, "del": 0x2E, "insert": 0x2D, "home": 0x24, "end": 0x23, "pageup": 0x21,
      "pagedown": 0x22, "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
      "volumeup": 0xAF, "volumedown": 0xAE, "mute": 0xAD, "playpause": 0xB3, "nexttrack": 0xB0,
      "prevtrack": 0xB1, "printscreen": 0x2C}
VK.update({f"f{i}": 0x6F + i for i in range(1, 25)})
VK.update({f"num{i}": 0x60 + i for i in range(10)})
VK.update({"pgup": 0x21, "pgdn": 0x22, "capslock": 0x14, "pause": 0x13, "scrolllock": 0x91, "menu": 0x5D,
           "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD, "\\": 0xDC, ";": 0xBA, "'": 0xDE, "`": 0xC0,
           ",": 0xBC, ".": 0xBE, "/": 0xBF})


def parse_keys(text):
    """'ctrl+shift+t' -> [vk, ...]; raises ValueError on an unknown key."""
    out = []
    for part in [x.strip().lower() for x in str(text or "").split("+") if x.strip()]:
        if part in VK:
            out.append(VK[part])
        elif len(part) == 1 and (part.isalpha() or part.isdigit()):
            out.append(ord(part.upper()))
        else:
            raise ValueError(f"unknown key {part!r}")
    if not out:
        raise ValueError("no keys")
    return out


def selftest():
    # landmarks: wrist (0) at bottom, middle MCP (9) 0.2 above -> hand size 0.2
    def hand(fingers=(True, True, True, True), thumb_to=None):
        p = [(0.5, 0.8)] * 21
        p[9] = (0.5, 0.6)
        xs = {8: 0.40, 12: 0.47, 16: 0.54, 20: 0.61}
        for (tip, up) in zip((8, 12, 16, 20), fingers):
            x = xs[tip]
            p[tip - 2] = (x, 0.55)                        # pip
            p[tip] = (x, 0.40 if up else 0.65)            # tip above / below pip
        p[4] = (0.25, 0.6) if thumb_to is None else (p[thumb_to][0] + 0.01, p[thumb_to][1])
        return p

    def hold(g, p, t, c=None, **kw):
        """Two frames 0.1 s apart (past the debounce) -> action kinds of both."""
        return [a[0] for a in g.step(p, t, c, **kw) + g.step(p, t + 0.1, c, **kw)]

    point, pinched = hand((True, False, False, False)), hand((True, False, False, False), thumb_to=8)
    g = Gestures({"pinch": 0.3, "palm_hold": 1.0, "scroll_speed": 1})
    kinds = lambda acts: [a[0] for a in acts]
    assert g.step(point, 0)[0][0] == "move" and g.mode == "move"
    # click and drag are separate: a quick pinch is one click with no button held
    gc = Gestures({"pinch": 0.3, "drag_hold": 0.35})
    assert not {"down", "click"} & set(kinds(gc.step(pinched, 0.1))), "one noisy frame never clicks"
    assert kinds(gc.step(pinched, 0.2)) == ["move"], "pinch registered; nothing fires before release"
    assert read_hand(point)["palm"] == read_hand(pinched)["palm"], "pinching never moves the aim point"
    acts = hold(gc, point, 0.3)
    assert acts.count("click") == 1 and "down" not in acts and not gc.down, acts
    # a held pinch drags: button down after drag_hold, up when the pinch opens
    hold(gc, pinched, 1.0)
    assert "down" not in kinds(gc.step(pinched, 1.3))
    acts = kinds(gc.step(pinched, 1.5))
    assert "down" in acts and gc.mode == "drag", acts
    assert "move" in kinds(gc.step(pinched, 1.55)), "a started drag moves the cursor"
    assert "up" not in kinds(gc.step(point, 1.6)), "one open frame never drops"
    assert "up" not in kinds(gc.step(pinched, 1.65)) and gc.down, "flicker back keeps the drag"
    acts = hold(gc, point, 1.7)
    assert "up" in acts and "click" not in acts and gc.mode == "move", acts
    # drag off: holding still just clicks on release
    gd = Gestures({"pinch": 0.3, "builtins": {"drag": False}})
    hold(gd, pinched, 0)
    assert "down" not in kinds(gd.step(pinched, 1.0)) and hold(gd, point, 1.1).count("click") == 1
    acts = hold(g, hand((True, True, False, False), thumb_to=12), 0.6)
    assert acts.count("right") == 1 and "down" not in acts, acts
    g.step(hand((True, True, False, False)), 0.8)
    g.step(hand((True, True, False, False)), 0.9)
    assert g.mode == "scroll"
    up = [(x, y - 0.1) for (x, y) in hand((True, True, False, False))]
    assert g.step(up, 1.0) == [("scroll", 3)]
    g.step(hand(), 1.1)
    assert ("pause",) in g.step(hand(), 2.2) and g.paused
    assert hold(g, pinched, 2.3) == [] and g.mode == "paused"
    g.step(hand(), 3.0)
    assert ("resume",) in g.step(hand(), 4.1) and not g.paused
    hold(g, pinched, 5.0)
    assert "down" in kinds(g.step(pinched, 5.5))
    assert g.step(None, 5.6) == [] and g.down, "a dropped frame never drops the drag"
    assert g.step(None, 5.8) == [("up",)], "hand really gone: let go"
    # right click: a held pinch fires once
    g4 = Gestures({"pinch": 0.3})
    rp = hand((True, True, False, False), thumb_to=12)
    assert sum(a[0] == "right" for t in range(20) for a in g4.step(rp, t * 0.05)) == 1
    # built-ins can be turned off
    g2 = Gestures({"builtins": {"click": False}})
    assert "down" not in hold(g2, pinched, 0)

    # two hands: the pointer keeps moving while the other hand pinches, and clicks come only from it
    g3 = Gestures({"pinch": 0.3})
    ptr, ptr_pinched = point, pinched
    acts = g3.step(ptr, 0.0, hand((False, False, False, False)))
    assert [a[0] for a in acts] == ["move"], acts
    assert hold(g3, ptr, 0.1, pinched) == ["move", "move"], "pinch registered, cursor keeps moving"
    acts = kinds(g3.step(ptr, 0.6, pinched))
    assert acts == ["move", "down"] and g3.mode == "drag", acts
    assert kinds(g3.step(ptr, 0.7, None, two=True)) == ["move"] and g3.down, "click hand lost: drag held"
    acts = hold(g3, ptr_pinched, 0.8, point)
    assert acts == ["move", "move", "up"], "pointer pinch never clicks; the click hand opened"
    assert hold(g3, ptr, 1.0, pinched) == ["move", "move"]
    assert hold(g3, ptr, 1.2, point) == ["move", "move", "click"], "quick pinch on the click hand = click"
    assert hold(g3, ptr, 1.6, hand((True, True, False, False), thumb_to=12)) == ["move", "move", "right"]

    # features: same pose moved, scaled and rolled -> ~same vector
    base = hand((True, True, False, False))
    def tf(p, ang, k, ox, oy):
        c, s = math.cos(ang), math.sin(ang)
        return [((x - 0.5) * k * c - (y - 0.8) * k * s + ox, (x - 0.5) * k * s + (y - 0.8) * k * c + oy) for x, y in p]
    assert pose_distance(features(base), features(tf(base, 0.5, 1.7, 0.3, 0.4))) < 0.01
    fist = hand((False, False, False, False))
    assert pose_distance(features(base), features(fist)) > 0.3

    # matcher: hold 0.4 s fires once, re-arms after the pose ends
    m = Matcher([{"id": "peace", "name": "Peace", "samples": [features(base)], "tolerance": 0.15, "hold": 0.4}])
    assert m.step(features(base), 0.0)[2] is None
    assert m.step(features(base), 0.5)[2]["id"] == "peace"
    assert m.step(features(base), 0.9)[2] is None
    assert m.step(features(fist), 1.0)[0] is None
    m.step(features(base), 1.1)
    assert m.step(features(base), 1.6)[2]["id"] == "peace"

    # two-hand gesture: matches only with both hands; a one-hand gesture still matches either hand alone
    both = features(base) + features(fist)
    m2 = Matcher([{"id": "combo", "samples": [both], "tolerance": 0.15, "hold": 0.0},
                  {"id": "peace", "samples": [features(base)], "tolerance": 0.15, "hold": 0.0}])
    assert m2.step([both, features(base), features(fist)], 0)[0]["id"] == "combo"
    assert m2.step([features(base)], 1)[0]["id"] == "peace"
    assert m2.step([features(fist)], 2)[0] is None

    # sticky roles: hands crossing or one dropping out never swaps move / click
    L = [(0.2, 0.8)] + [(0.2, 0.6)] * 20
    R = [(0.8, 0.8)] + [(0.8, 0.6)] * 20
    mem = {}
    ptr, clk, two = assign_hands([L, R], True, mem, 0.0)
    assert ptr is R and clk is L and two
    R2 = [(0.45, 0.8)] + R[1:]
    L2 = [(0.4, 0.8)] + L[1:]
    ptr, clk, _ = assign_hands([L2, R2], True, mem, 0.1)
    assert ptr is R2, "close hands keep their roles"
    assert assign_hands([L2], True, mem, 0.2) == (None, None, True), "click hand alone never moves the cursor"
    assert assign_hands([R2], True, mem, 0.3)[:2] == (R2, None), "pointer alone keeps moving"
    assert assign_hands([L2], True, mem, 2.0) == (L2, None, False), "after the grace: plain one-hand mode"

    # preview: only one hand is highlighted unless a two-hand gesture / recording uses both
    a, b = hand(), fist
    act = lambda looks: [l[2] for l in looks]
    assert act(hand_looks([a, b], b, True, "move", None, None, False, False)) == [False, True]
    assert act(hand_looks([a, b], b, True, "drag", None, None, True, True)) == [True, False]
    assert act(hand_looks([a, b], b, True, "gesture", "x", 2, False, False)) == [True, False]
    assert act(hand_looks([a, b], b, True, "gesture", "x", 1, False, False)) == [False, True]
    assert act(hand_looks([a, b], b, True, "gesture", "x", 0, False, False)) == [True, True]
    assert act(hand_looks([a, b], b, False, "move", None, None, False, False)) == [False, True]

    # both hands pinched: one two-finger touch from start to end, no click or drag around it
    gt = Gestures({"pinch": 0.3})
    acts = hold(gt, pinched, 0.0, pinched)
    assert acts == ["move", "touch_down"] and gt.pinch_at is None and not gt.down, acts
    assert kinds(gt.step(pinched, 0.2, pinched)) == ["touch_move"] and gt.mode == "touch"
    assert kinds(gt.step(pinched, 0.25, None, two=True)) == ["touch_hold"], "click hand lost: fingers stay"
    acts = hold(gt, point, 0.3, pinched)
    assert acts == ["touch_move", "touch_up", "move"], acts
    assert hold(gt, point, 1.0, pinched) == ["move", "move"] and gt.pinch_at is None, "still-pinched hand never clicks"
    hold(gt, point, 1.5, point)
    assert hold(gt, point, 2.0, pinched) == ["move", "move"] and gt.pinch_at is not None, "re-armed after opening"
    gt2 = Gestures({"pinch": 0.3})
    hold(gt2, pinched, 0.0, pinched)
    assert gt2.step(None, 0.3) == [("touch_hold",)], "a dropped frame keeps the fingers down"
    assert gt2.step(None, 0.5) == [("touch_up",)], "hands gone: fingers lift"
    assert "touch_down" not in hold(Gestures({"pinch": 0.3, "builtins": {"pinch2": False}}), pinched, 0, pinched)
    # touch geometry: starts TOUCH_GAP apart on the cursor, spreads, turns and pans with the hands
    A = {"cx": 500, "cy": 400, "mx": 0.5, "my": 0.5, "len0": 0.2 * 640, "kx": 1000, "ky": 1000}
    (x1, y1), (x2, y2) = touch_geometry((0.4, 0.5), (0.6, 0.5), A, 640, 480)
    assert abs(x1 - 400) < 1e-6 and abs(x2 - 600) < 1e-6 and y1 == y2 == 400
    (x1, _), (x2, _) = touch_geometry((0.3, 0.5), (0.7, 0.5), A, 640, 480)
    assert abs((x2 - x1) - 2 * TOUCH_GAP) < 1e-6, "hands twice as far apart = fingers twice as far apart"
    q = 0.1 * 640 / 480  # same 128 px gap as the start, but vertical
    (x1, y1), (x2, y2) = touch_geometry((0.5, 0.5 - q), (0.5, 0.5 + q), A, 640, 480)
    assert abs(x1 - x2) < 1e-6 and abs((y2 - y1) - TOUCH_GAP) < 1e-6, "turned a quarter: fingers turn with the hands"
    (x1, _), (x2, _) = touch_geometry((0.5, 0.5), (0.7, 0.5), A, 640, 480)
    assert abs((x1 + x2) / 2 - 600) < 1e-6, "midpoint moved right by 0.1 -> 100 px pan"
    assert ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(POINTER_TOUCH_INFO) == 144
    assert ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(INPUT) == 40

    # game: stick hand -> direction keys with a deadzone and sticky edge; look hand -> mouse motion
    gm = Game()
    rest = hand()
    assert gm.stick(rest, 0.35) == set()
    shift = lambda p, dx, dy: [(x + dx, y + dy) for x, y in p]
    assert gm.stick(shift(rest, 0, -0.1), 0.35) == {"up"}, "palm up half a hand = forward"
    assert gm.stick(shift(rest, 0.1, 0), 0.35) == {"right"}
    assert gm.stick(shift(rest, 0.06, 0), 0.35) == {"right"}, "inside 70% of the deadzone lets go, not before"
    assert gm.stick(shift(rest, 0.04, 0), 0.35) == set()
    assert gm.stick(shift(rest, -0.1, 0.1), 0.35) == {"left", "down"}, "diagonals hold two keys"
    assert gm.stick(None, 0.35) == set() and gm.center is None, "hand gone: keys up, recenter next time"
    gm.stick(rest, 0.35, 1.0)
    assert gm.stick(shift(rest, 0.1, 0), 0.35, 1.03) == {"right"}
    assert gm.stick(None, 0.35, 1.1) == {"right"} and gm.center is not None, "a dropped frame keeps walking"
    assert gm.stick(None, 0.35, 1.3) == set() and gm.center is None
    assert gm.pinches("look", pinched, 0.3, 0.48, 1.0) == {"pinch"}
    assert gm.pinches("look", None, 0.3, 0.48, 1.05) == {"pinch"}, "a dropped frame keeps the button down"
    assert gm.pinches("look", None, 0.3, 0.48, 1.3) == set()
    assert gm.pinches("look", pinched, 0.3, 0.48) == {"pinch"}
    assert gm.pinches("look", point, 0.3, 0.48) == set()
    assert gm.look(rest, 0.0, 4000) == (0, 0)
    mx, my = gm.look(shift(rest, 0.01, 0), 0.033, 4000)
    assert mx > 0 and my == 0, (mx, my)
    assert gm.look(None, 0.1, 4000) == (0, 0) and gm.prev is None

    assert parse_keys("Ctrl+Shift+T") == [0x11, 0x10, ord("T")]
    # every name the panel's key capture can send is known
    for k in ("pgup", "pgdn", "playpause", "volumeup", "nexttrack", "num5", "f13", "/", "capslock", "menu", "win"):
        parse_keys("ctrl+" + k)
    try:
        parse_keys("ctrl+banana")
        raise AssertionError("bad key accepted")
    except ValueError:
        pass
    # one euro: jitter at rest is damped, a fast move is followed closely
    import random
    rnd = random.Random(1)
    f = OneEuro(1.0, 8.0)
    outs = [f(0.5 + rnd.uniform(-0.01, 0.01), i / 30) for i in range(60)]
    assert max(outs[30:]) - min(outs[30:]) < 0.01, "still hand should not shake"
    for i in range(60, 66):
        last = f(0.9, i / 30)
    assert last > 0.8, last

    # latest-frame reader skips stale frames instead of queueing them
    class FakeCap:
        i = 0
        def read(self):
            time.sleep(0.005)
            self.i += 1
            return True, self.i
        def release(self):
            pass
    lf = LatestFrame(FakeCap())
    ok, f1, n1 = lf.read(0)
    time.sleep(0.1)
    ok2, f2, n2 = lf.read(n1)
    lf.close()
    assert ok and ok2 and f2 > f1 + 3, (f1, f2)
    assert lf.seen_t > 0, "frames carry their arrival time"

    # mover: the cursor glides between camera frames, leads a moving hand, never passes a still one
    sent, rels = [], []
    mv = Mover(lambda x, y: sent.append((x, y)), lambda dx, dy: rels.append((dx, dy)), start=False)
    mv.aim(100.0, 100.0, 0.0, 0.0)
    mv.aim(133.0, 100.0, 0.033, 0.0)
    for i in range(1, 9):
        mv.tick(0.033 + i * 0.004, 0.004)
    xs = [x for x, _ in sent]
    assert len(xs) > 3 and xs == sorted(xs) and xs[-1] in (132, 133), xs
    mv.aim(166.0, 100.0, 0.066, 1.0)  # still moving at 1000 px/s: lead it
    for i in range(1, 9):
        mv.tick(0.066 + i * 0.004, 0.004)
    assert 166 < sent[-1][0] <= 166 + 33, sent[-1]
    mv.aim(166.0, 100.0, 0.099, 1.0)  # stopped: settle on the point, no lead
    for i in range(1, 12):
        mv.tick(0.099 + i * 0.004, 0.004)
    assert sent[-1] == (166, 100), sent[-1]
    mv.idle()
    n = len(sent)
    mv.tick(0.2, 0.004)
    assert len(sent) == n, "idle: the real mouse owns the cursor"
    mv.push(10, -3)
    for i in range(20):
        mv.tick(0.3 + i * 0.004, 0.004)
    assert len(rels) > 2 and (sum(r[0] for r in rels), sum(r[1] for r in rels)) == (10, -3), rels
    # sticky pixel: resting between two pixels never flickers; touch fingers glide, then stop
    mv.aim(200.5, 50.0, 0.5, 0.0)
    mv.tick(0.504, 0.004)
    n = len(sent)
    for i, x in enumerate((200.4, 200.6, 200.45, 200.55)):
        mv.aim(x, 50.0, 0.533 + i * 0.033, 0.0)
        for j in range(8):
            mv.tick(0.533 + i * 0.033 + j * 0.004, 0.004)
    assert len(sent) == n, sent[n:]
    fing = []
    mt = Mover(lambda x, y: None, lambda dx, dy: None, fing.append, start=False)
    mt.touch_to([(0, 0), (100, 0)])
    mt.tick(1.0, 0.004)
    mt.touch_to([(40, 0), (140, 0)])
    for i in range(1, 30):
        mt.tick(1.0 + i * 0.004, 0.004)
    a = [f[0][0] for f in fing]
    assert a[0] == 0 and a[-1] == 40 and a == sorted(a) and len(set(a)) > 4, a
    mt.touch_off()
    n = len(fing)
    mt.tick(2.0, 0.004)
    assert len(fing) == n
    # landmark smoother: a shaking hand is steadied, a moving one is followed, two hands never mix
    hs = HandSmoother()
    base = hand()
    outs = [hs([[(x + rnd.uniform(-0.004, 0.004), y + rnd.uniform(-0.004, 0.004)) for x, y in base]], i / 30)[0][8][0]
            for i in range(60)]
    assert max(outs[30:]) - min(outs[30:]) < 0.004, "smoothed landmark shakes less than the raw one"
    far = [(x + 0.3, y) for x, y in base]
    for i in range(60, 66):
        two_out = hs([[(x + (i - 59) * 0.02, y) for x, y in base], far], i / 30)
    assert abs(two_out[0][0][0] - (0.5 + 6 * 0.02)) < 0.02 and abs(two_out[1][0][0] - 0.8) < 1e-6, two_out[0][0]
    print("selftest ok")


# ---------- Windows input ----------

LEFTDOWN, LEFTUP, RIGHTDOWN, RIGHTUP, MIDDOWN, MIDUP, WHEEL = 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x800
KEYUP = 0x2
MOUSE_BUTTONS = {"lmb": (LEFTDOWN, LEFTUP), "rmb": (RIGHTDOWN, RIGHTUP), "mmb": (MIDDOWN, MIDUP)}
EXTENDED_VK = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5B, 0x5D}


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_size_t)]


class _INPUTU(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTU)]


def parent_alive(pid):
    if not pid:
        return True
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
    if not h:
        return False
    try:
        return k.WaitForSingleObject(h, 0) != 0  # 0 = signaled = exited
    finally:
        k.CloseHandle(h)


# preview colors (BGR), matching the panel's mode colors
MODE_BGR = {"move": (248, 189, 56), "drag": (94, 197, 34), "scroll": (11, 158, 245), "gesture": (247, 85, 168),
            "recording": (68, 68, 239), "paused": (175, 163, 156), "nohand": (175, 163, 156), "touch": (21, 204, 250)}
ROLE_BGR = {"pointer": (248, 189, 56), "click": (94, 197, 34)}
MODE_LABEL = {"move": "MOVE", "drag": "DRAG", "scroll": "SCROLL", "paused": "PAUSED", "recording": "REC", "touch": "PINCH"}


def hand_looks(hands, pointer, two, mode, gesture_name, gesture_idx, click_busy, down):
    """-> [(points, color, active, label)] per hand. Exactly one hand is 'active' (bright, the one
    whose gesture is in effect); both only for a two-hand gesture or while recording."""
    out = []
    for p in hands:
        is_ptr = p is pointer
        dim = (ROLE_BGR["pointer"] if is_ptr else ROLE_BGR["click"], False, "MOVE HAND" if is_ptr else "CLICK HAND")
        if mode == "recording":
            look = (MODE_BGR["recording"], True, "REC")
        elif mode == "touch":
            look = (MODE_BGR["touch"], True, "PINCH")  # both hands make the touch
        elif mode == "gesture":
            # matcher candidates: 0 = both hands, 1 = pointer hand, 2 = other hand (one hand seen: 0 = it)
            mine = len(hands) == 1 or gesture_idx == 0 or (gesture_idx == 1) == is_ptr
            look = (MODE_BGR["gesture"], True, (gesture_name or "GESTURE").upper()[:18]) if mine else dim
        elif not is_ptr and not two:
            look = ((120, 120, 120), False, "IGNORED")  # two-hand mode off: the second hand does nothing
        elif not is_ptr:
            look = (MODE_BGR["drag"], True, "DRAG" if down else "CLICK") if click_busy else dim
        elif two and click_busy:
            look = dim
        else:
            look = (MODE_BGR.get(mode, ROLE_BGR["pointer"]), True, MODE_LABEL.get(mode, "MOVE"))
        out.append((p,) + look)
    return out


def draw_hand(cv2, frame, conns, p, color, active, label):
    h, w = frame.shape[:2]
    xy = [(int(x * w), int(y * h)) for x, y in p]
    col = color if active else tuple(int(c * 0.5) for c in color)
    for a, b in conns:
        cv2.line(frame, xy[a], xy[b], col, 4 if active else 2, cv2.LINE_AA)
    for i, pt in enumerate(xy):
        cv2.circle(frame, pt, 7 if active and i in (4, 8, 12) else 3, col, -1, cv2.LINE_AA)
    x0, y0 = min(x for x, _ in xy), max(22, min(y for _, y in xy) - 6)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x0, y0 - th - 10), (x0 + tw + 10, y0), col, -1)
    cv2.putText(frame, label, (x0 + 5, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (24, 24, 24), 2, cv2.LINE_AA)


class LatestFrame:
    """Reads the camera on its own thread and keeps only the newest frame, so slow processing
    never falls behind a queue of stale frames (that queue is what makes video lag)."""

    def __init__(self, cap):
        self.cap, self.ok, self.frame, self.n, self.alive = cap, True, None, 0, True
        self.t = self.seen_t = 0.0  # perf_counter time the newest / the last returned frame arrived
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.alive:
            ok, f = self.cap.read()
            with self.cond:
                self.ok = ok
                if ok:
                    self.frame, self.n, self.t = f, self.n + 1, time.perf_counter()
                self.cond.notify_all()
            if not ok:
                time.sleep(0.05)

    def read(self, seen):
        """-> (ok, frame, n): waits up to 1 s for a frame newer than `seen`."""
        with self.cond:
            self.cond.wait_for(lambda: self.n != seen or not self.ok or not self.alive, timeout=1.0)
            fresh = self.n != seen
            self.seen_t = self.t
            return fresh, self.frame if fresh else None, self.n

    def close(self):
        self.alive = False
        self.thread.join(timeout=2)
        self.cap.release()


def glide(pos, target, vel, age, dt, predict, tau=0.008):
    """One output tick between camera frames: aim a little ahead of the last measured point
    (its speed * its age * predict), then ease toward that. predict 0 = no lead. -> new pos"""
    lead = min(max(age, 0.0), 0.05) * predict
    k = 1.0 - math.exp(-dt / tau)
    return (pos[0] + (target[0] + vel[0] * lead - pos[0]) * k,
            pos[1] + (target[1] + vel[1] * lead - pos[1]) * k)


class Mover:
    """The camera gives 30-60 points a second. This thread sends the cursor (and the game camera)
    ~240 times a second in between, so motion is smooth and slightly ahead instead of stepping.
    set_pos(x, y) = absolute cursor, move_rel(dx, dy) = relative mouse (game look)."""

    def __init__(self, set_pos, move_rel, move_touch=None, hz=240, start=True):
        self.set_pos, self.move_rel, self.move_touch, self.dt = set_pos, move_rel, move_touch, 1.0 / hz
        self.lock, self.out = threading.Lock(), threading.Lock()
        self.t_target = self.t_pos = self.t_sent = None  # the two touch fingers
        self.t_at = 0.0
        self.target = self.pos = self.sent = None
        self.vel, self.t, self.predict = (0.0, 0.0), 0.0, 0.5
        self.owed = [0.0, 0.0]
        self.ticks, self.alive = 0, True
        self.thread = threading.Thread(target=self._run, daemon=True)
        if start:
            self.thread.start()

    def aim(self, x, y, t, predict=0.5):
        """A new measured cursor point in pixels, seen at perf_counter time t."""
        with self.lock:
            if self.target is not None and 0 < t - self.t < 0.2:
                self.vel = ((x - self.target[0]) / (t - self.t), (y - self.target[1]) / (t - self.t))
            else:
                self.vel = (0.0, 0.0)
            if self.pos is None:
                self.pos = (x, y)
            self.target, self.t, self.predict = (x, y), t, predict

    def idle(self):
        """Nothing is steering the cursor this frame: leave it alone (the real mouse owns it)."""
        with self.lock:
            self.target = self.pos = self.sent = None

    def push(self, dx, dy):
        """Relative mouse movement to send, spread over the next few ticks."""
        with self.lock:
            self.owed[0] += dx
            self.owed[1] += dy

    def drop(self):
        with self.lock:
            self.owed = [0.0, 0.0]

    def touch_to(self, pts):
        """Where the two touch fingers should be (screen px); they glide there."""
        with self.lock:
            self.t_target = [tuple(q) for q in pts]
            if self.t_pos is None:
                self.t_pos = list(self.t_target)

    def touch_off(self):
        """Stop sending touch moves. Returns only once no move is in flight."""
        with self.out:
            with self.lock:
                self.t_target = self.t_pos = self.t_sent = None

    def tick(self, now, dt):
        with self.out:
            self._tick(now, dt)

    def _tick(self, now, dt):
        with self.lock:
            self.ticks += 1
            out = rel = fingers = None
            if self.t_target is not None:
                k = 1.0 - math.exp(-dt / 0.012)
                self.t_pos = [(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
                              for a, b in zip(self.t_pos, self.t_target)]
                f = [(int(round(x)), int(round(y))) for x, y in self.t_pos]
                if f != self.t_sent or now - self.t_at > 0.05:  # Windows drops contacts that go quiet
                    self.t_sent, self.t_at, fingers = f, now, f
            if self.target is not None:
                self.pos = glide(self.pos, self.target, self.vel, now - self.t, dt, self.predict)
                p = (int(round(self.pos[0])), int(round(self.pos[1])))
                # sticky pixel: a point resting between two pixels must not flicker between them
                if self.sent is None or max(abs(self.pos[0] - self.sent[0]), abs(self.pos[1] - self.sent[1])) >= 0.75:
                    self.sent = out = p
            if self.owed[0] or self.owed[1]:
                k = 1.0 - math.exp(-dt / 0.008)
                step = []
                for i in (0, 1):
                    s = self.owed[i] * k
                    m = int(s) if abs(s) >= 1 else int(self.owed[i])  # the last pixels go out whole
                    self.owed[i] -= m
                    step.append(m)
                if step[0] or step[1]:
                    rel = step
        if out:
            self.set_pos(*out)
        if rel:
            self.move_rel(*rel)
        if fingers and self.move_touch:
            self.move_touch(fingers)

    def _run(self):
        last = time.perf_counter()
        while self.alive:
            time.sleep(self.dt)
            now = time.perf_counter()
            self.tick(now, now - last)
            last = now

    def close(self):
        self.alive = False
        if self.thread.is_alive():
            self.thread.join(timeout=1)


class Overlay:
    """Click-through ring that follows the cursor and shows the mode."""
    KEY = "#010203"
    COLORS = {"move": "#38bdf8", "drag": "#22c55e", "scroll": "#f59e0b", "gesture": "#a855f7",
              "recording": "#ef4444", "paused": "#9ca3af", "nohand": "#9ca3af", "touch": "#facc15"}

    def __init__(self, user32):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", self.KEY)
        self.root.config(bg=self.KEY)
        self.c = tk.Canvas(self.root, width=52, height=52, bg=self.KEY, highlightthickness=0)
        self.c.pack()
        self.ring = self.c.create_oval(5, 5, 47, 47, outline=self.COLORS["move"], width=4)
        self.dot = self.c.create_oval(22, 22, 30, 30, fill=self.KEY, outline="")
        self.root.update()
        hwnd = int(self.root.wm_frame(), 16)
        ex = user32.GetWindowLongW(hwnd, -20)
        # layered | transparent (click-through) | toolwindow (no taskbar) | noactivate
        user32.SetWindowLongW(hwnd, -20, ex | 0x80000 | 0x20 | 0x80 | 0x08000000)
        self.mode = None

    def hide(self):
        if self.mode != "hidden":
            self.root.withdraw()
            self.root.update()
            self.mode = "hidden"

    def show(self, x, y, mode):
        if self.mode == "hidden":
            self.root.deiconify()
            self.mode = None
        if mode != self.mode:
            col = self.COLORS.get(mode, self.COLORS["move"])
            self.c.itemconfig(self.ring, outline=col)
            self.c.itemconfig(self.dot, fill=col if mode in ("drag", "recording") else self.KEY)
            self.mode = mode
        self.root.geometry(f"+{int(x) - 26}+{int(y) - 26}")
        self.root.update()

    def close(self):
        try:
            self.root.destroy()
        except Exception:
            pass


def main(folder, parent):
    import cv2
    import mediapipe as mp

    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    try:  # 1 ms sleeps (the 240 Hz cursor thread needs them) and a step above normal priority
        ctypes.windll.winmm.timeBeginPeriod(1)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x8000)
    except Exception:
        pass
    sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)  # ponytail: primary monitor only
    P = lambda name: os.path.join(folder, name)

    def load_json(path, default):
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return default

    def write_atomic(path, data, binary=False):
        tmp = path + ".tmp"
        with open(tmp, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as fh:
            fh.write(data)
        # Windows refuses the swap while the panel is reading the old file: retry, then skip this write
        for _ in range(5):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:
                time.sleep(0.01)
        return False

    events = collections.deque(maxlen=15)

    def event(text):
        events.appendleft({"t": round(time.time(), 2), "text": text})

    def click(down, up, times=1):
        for _ in range(times):
            user32.mouse_event(down, 0, 0, 0, 0)
            user32.mouse_event(up, 0, 0, 0, 0)

    def wheel(n):
        user32.mouse_event(WHEEL, 0, 0, ctypes.c_uint32(n * 120 & 0xFFFFFFFF).value, 0)

    def press(keys):
        vks = parse_keys(keys)
        for vk in vks:
            user32.keybd_event(vk, 0, 0, 0)
        for vk in reversed(vks):
            user32.keybd_event(vk, 0, KEYUP, 0)

    user32.MapVirtualKeyW.restype = ctypes.c_uint
    user32.GetAsyncKeyState.restype = ctypes.c_short

    def send_key(name, up):
        """Hold (up=False) or let go of a key, combo or mouse button ('lmb' / 'rmb' / 'mmb').
        Keys go out as scan codes: games that read raw input see them like a real keyboard."""
        if name in MOUSE_BUTTONS:
            user32.mouse_event(MOUSE_BUTTONS[name][1 if up else 0], 0, 0, 0, 0)
            return
        vks = parse_keys(name)
        for vk in (reversed(vks) if up else vks):
            inp = INPUT(type=1)
            inp.u.ki = KEYBDINPUT(0, user32.MapVirtualKeyW(vk, 0),
                                  0x8 | (0x1 if vk in EXTENDED_VK else 0) | (KEYUP if up else 0), 0, 0)
            user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

    def undo(acts):
        """Undo what Gestures.release() hands back: let go of the button, lift touch fingers."""
        for a in acts:
            if a[0] == "up":
                user32.mouse_event(LEFTUP, 0, 0, 0, 0)
            elif a[0] == "touch_up":
                mover.touch_off()
                touch.up()

    held_gesture = {}  # a custom gesture holding its keys down while the pose lasts: {id, keys}

    def let_go_gesture():
        if held_gesture:
            try:
                send_key(held_gesture["keys"], True)
            except ValueError:
                pass
            held_gesture.clear()

    def run_action(g):
        a = g.get("action") or {}
        kind = a.get("type", "none")
        if kind == "keys" and a.get("hold"):
            let_go_gesture()
            send_key(a.get("keys", ""), False)
            held_gesture.update(id=g.get("id"), keys=a.get("keys", ""))
            event(f"Gesture '{g.get('name', g.get('id'))}' holding {a.get('keys', '')}")
            return
        if kind == "click":
            click(LEFTDOWN, LEFTUP)
        elif kind == "double_click":
            click(LEFTDOWN, LEFTUP, 2)
        elif kind == "right_click":
            click(RIGHTDOWN, RIGHTUP)
        elif kind == "middle_click":
            click(MIDDOWN, MIDUP)
        elif kind == "scroll":
            wheel(int(a.get("amount", 3)))
        elif kind == "keys":
            press(a.get("keys", ""))
        elif kind == "pause":
            gest.paused = not gest.paused
        label = {"keys": a.get("keys", "")}.get(kind, kind.replace("_", " "))
        event(f"Gesture '{g.get('name', g.get('id'))}' -> {label}")

    cfg = load_json(P("config.json"), {})
    cam_index = int(cfg.get("camera", 0))
    cam_name = str(cfg.get("camera_name") or f"Camera {cam_index}")
    bad_since = dark_since = None

    def no_picture(secs):
        if secs < 4:
            return ""
        hint = (" The phone camera streams to one app at a time: close camera tabs in Chrome/Edge or press 'Reset phone camera'."
                if "Windows Virtual Camera" in cam_name else
                " Start the app that feeds it (OBS, Quest Link…) or pick another camera in Settings.")
        return f"'{cam_name}' isn't sending a picture.{hint}"
    def go_fast(c):
        """Ask a real webcam for 640x480 MJPG at 60 fps: twice the points, half the wait. -> still has a picture"""
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        c.set(cv2.CAP_PROP_FPS, 60)
        t0 = time.time()
        while time.time() - t0 < 1.5:
            if c.read()[0]:
                return True
            time.sleep(0.03)
        return False

    def open_camera(fast):
        # Media Foundation first (what Chrome uses; the only path that gets a picture from
        # Phone Link / Windows Virtual Camera), DirectShow as the fallback.
        for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
            c = cv2.VideoCapture(cam_index, api)
            if api == cv2.CAP_DSHOW:
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            t0 = time.time()
            while c.isOpened() and time.time() - t0 < 3:
                if c.read()[0]:
                    name = "Media Foundation" if api == cv2.CAP_MSMF else "DirectShow"
                    if not fast:
                        return c, name
                    if go_fast(c):
                        return c, name + ", 60 fps asked"
                    c.release()  # the camera lost its picture in that mode: open it plain
                    return open_camera(False)
                time.sleep(0.05)
            c.release()
        return cv2.VideoCapture(cam_index, cv2.CAP_DSHOW), "DirectShow"  # opens; loop reports no picture

    # virtual cameras (phone, OBS, headsets) are left as they are: changing their mode can drop the stream
    cap, cam_api = open_camera(bool(cfg.get("fast_camera", True)) and not cfg.get("camera_virtual", True))
    if not cap.isOpened():
        write_atomic(P("status.json"), json.dumps({"mode": "error", "error": f"'{cam_name}' did not open. Another app may be using it, or pick another camera in Settings."}))
        sys.exit(2)
    grab = LatestFrame(cap)
    seen = 0
    hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0,
                                     min_detection_confidence=0.6, min_tracking_confidence=0.5)
    conns = mp.solutions.hands.HAND_CONNECTIONS
    gest = Gestures(cfg)
    matcher = Matcher(cfg.get("gestures"))
    touch = Touch(user32)
    smooth = HandSmoother()
    mover = Mover(lambda x, y: user32.SetCursorPos(x, y),
                  lambda dx, dy: user32.mouse_event(1, dx, dy, 0, 0),  # relative: works with locked game cursors
                  lambda pts: touch.move(pts))
    ms = hz = 0.0
    hz_ticks = 0
    touch_anchor = {}
    tf = [OneEuro() for _ in range(4)]  # the two touch points, x / y each
    game = cfg_game = bool(cfg.get("game_mode"))
    gm = Game()
    game_held = set()
    hotkey_was = False

    def release_game_keys():  # never leave a game key stuck down
        for k in sorted(game_held):
            try:
                send_key(k, True)
            except ValueError:
                pass
        game_held.clear()

    def set_game(on, why=""):
        nonlocal game
        if on == game:
            return
        game = on
        undo(gest.release())
        release_game_keys()
        gm.reset()
        mover.drop()
        event(("Game mode on" if on else "Game mode off") + why)
    def make_overlay():
        try:
            return Overlay(user32)
        except Exception as exc:  # tracking still works without the ring
            event(f"Cursor ring unavailable: {str(exc).splitlines()[0][:120]}")
            return None

    overlay = make_overlay() if cfg.get("overlay", True) else None
    overlay_failed = cfg.get("overlay", True) and overlay is None
    rec = None  # {id, name, phase, until, samples}
    fx, fy = OneEuro(), OneEuro()
    roles = {}  # sticky two-hand roles (assign_hands)
    match_info = None
    last_cfg = last_preview = last_status = time.time()
    frames, fps, fps_t = 0, 0.0, time.time()
    event(f"Started on '{cam_name}' ({cam_api})")
    try:
        while True:
            now = time.time()
            if os.path.exists(P("stop")) or not parent_alive(parent):
                break
            if now - last_cfg > 0.5:
                cfg = load_json(P("config.json"), cfg)
                gest.cfg = cfg
                matcher.set(cfg.get("gestures"))
                if bool(cfg.get("game_mode")) != cfg_game:  # the panel switched it
                    cfg_game = bool(cfg.get("game_mode"))
                    set_game(cfg_game)
                want = bool(cfg.get("overlay", True))
                if not want:
                    overlay_failed = False
                if want != (overlay is not None) and not overlay_failed:
                    if overlay:
                        overlay.close()
                        overlay = None
                    else:
                        overlay = make_overlay()
                        overlay_failed = overlay is None
                if os.path.exists(P("record_cancel")):
                    try:
                        os.remove(P("record_cancel"))
                    except OSError:
                        pass
                    if rec is not None:
                        rec = None
                        event("Recording cancelled")
                req = load_json(P("record.json"), None)
                if req and rec is None:
                    os.remove(P("record.json"))
                    undo(gest.release())
                    let_go_gesture()
                    release_game_keys()
                    gm.reset()
                    rec = {"id": req["id"], "name": req.get("name", ""), "phase": "countdown",
                           "until": now + float(req.get("countdown", 3)), "seconds": float(req.get("seconds", 2)),
                           "samples": []}
                    event(f"Recording '{rec['name']}': get ready")
                last_cfg = now
            ok, frame, seen = grab.read(seen)
            if not ok:
                bad_since = bad_since or now
                if now - last_status > 0.5:
                    write_atomic(P("status.json"), json.dumps({
                        "mode": "nocamera", "fps": 0, "hand": False, "t": now, "events": list(events),
                        "warning": no_picture(now - bad_since)}))
                    last_status = now
                time.sleep(0.05)
                continue
            bad_since = None
            ft = grab.seen_t  # when this frame arrived (perf_counter): what the filters and the glide run on
            aimed = False
            # a virtual camera that isn't connected (phone, OBS, headset) sends black placeholder frames
            if float(frame[::8, ::8].mean()) < 14:
                dark_since = dark_since or now
            else:
                if dark_since and now - dark_since > 4:
                    event(f"Camera '{cam_name}' picture is back")
                dark_since = None
            if frame.shape[1] > 640:  # MSMF delivers the camera's native size (720p+)
                frame = cv2.resize(frame, (640, frame.shape[0] * 640 // frame.shape[1]))
            frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lms = list(res.multi_hand_landmarks or [])[:2]
            lm = lms[0] if lms else None
            # left to right on screen (the image is mirrored, so screen right = your right hand)
            seen_hands = sorted(smooth([[(q.x, q.y) for q in l.landmark] for l in lms], ft), key=lambda p: p[0][0])
            ptr_right = cfg.get("pointer_hand", "right") != "left"
            if cfg.get("two_hands", True):
                pts, click_pts, two = assign_hands(seen_hands, ptr_right, roles, now)
            else:
                pts, click_pts, two = ((seen_hands[-1] if ptr_right else seen_hands[0]) if seen_hands else None), None, False
            # candidate fingerprints: both hands together (2-hand gestures), then each hand alone
            if len(seen_hands) == 2:
                other = seen_hands[0] if pts is seen_hands[1] else seen_hands[1]
                feats = [features(seen_hands[0]) + features(seen_hands[1]), features(pts), features(other)]
            else:
                feats = [features(seen_hands[0])] if seen_hands else []
            feat = feats[0] if feats else None
            mode = None
            look_p = stick_p = None
            hotkey = all(user32.GetAsyncKeyState(k) & 0x8000 for k in (0x11, 0x12, 0x47))  # Ctrl+Alt+G
            if hotkey and not hotkey_was:
                set_game(not game, " (Ctrl+Alt+G)")
                write_atomic(P("game_set.json"), json.dumps({"game": game}))  # the backend saves it
            hotkey_was = hotkey

            if rec is not None:
                # recording: no mouse actions, collect the pose
                mode = "recording"
                if rec["phase"] == "countdown" and now >= rec["until"]:
                    rec["phase"], rec["until"] = "capturing", now + rec["seconds"]
                elif rec["phase"] == "capturing":
                    if feat:
                        rec["samples"].append(feat)
                    if now >= rec["until"]:
                        # keep the hand count seen most (a second hand that flickered in or out is dropped)
                        lens = collections.Counter(len(s) for s in rec["samples"])
                        size = lens.most_common(1)[0][0] if lens else 0
                        kept = [s for s in rec["samples"] if len(s) == size]
                        samples = kept[:: max(1, len(kept) // 40)][:40]
                        result = {"id": rec["id"], "name": rec["name"], "samples": samples, "hands": size // 40}
                        if len(samples) < 5:
                            result = {"id": rec["id"], "error": "Hand not seen enough while recording. Keep it in view and try again."}
                        write_atomic(P("record_result.json"), json.dumps(result))
                        event(f"Recorded '{rec['name']}' ({len(samples)} samples)" if "samples" in result else "Recording failed: hand not seen")
                        rec = None
            else:
                hit, dist, fire = matcher.step(feats, now)
                match_info = {"name": hit.get("name"), "id": hit["id"], "distance": round(dist, 3)} if hit else (
                    {"name": None, "distance": round(dist, 3)} if feat else None)
                if held_gesture and (hit is None or hit["id"] != held_gesture.get("id")):
                    let_go_gesture()  # the pose ended: let go of its held keys
                if fire:
                    try:
                        run_action(fire)
                    except Exception as exc:
                        event(f"Gesture '{fire.get('name')}' failed: {exc}")
                if game:
                    # stick hand = direction keys, look hand = camera; both hands' pinches = buttons
                    mode = "gesture" if hit else "game"
                    if len(seen_hands) == 2:
                        look_p, stick_p, _ = assign_hands(seen_hands, ptr_right, roles, now)
                    elif seen_hands:
                        one = seen_hands[0]
                        if roles.get("t") is not None and now - roles["t"] < 0.6:
                            is_look = _d(one[0], roles["ptr"]) <= _d(one[0], roles["click"])
                        else:
                            is_look = (one[0][0] >= 0.5) == ptr_right
                        look_p, stick_p = (one, None) if is_look else (None, one)
                    keys = {**GAME_KEYS, **(cfg.get("game_keys") or {})}
                    on_p = float(cfg.get("pinch", 0.35))
                    names = ([d for d in gm.stick(stick_p, float(cfg.get("stick_dead", 0.35)), now)]
                             + list(gm.pinches("stick", stick_p, on_p, on_p * 1.6, now))
                             + ["look_" + k for k in gm.pinches("look", look_p, on_p, on_p * 1.6, now)])
                    want = {keys.get(n) for n in names} - {"", None}
                    for k in sorted(game_held - want) + sorted(want - game_held):
                        try:
                            send_key(k, k not in want)
                        except ValueError:
                            event(f"Game key '{k}' isn't a key")
                            want.discard(k)
                    game_held.clear()
                    game_held.update(want)
                    mx, my = gm.look(look_p, ft, float(cfg.get("look_speed", 4)) * sw * 0.4)
                    if mx or my:
                        mover.push(mx, my)  # sent as many small steps: a smooth camera, not 30 jumps a second
                elif hit and not gest.paused:
                    undo(gest.release())
                    mode = "gesture"
                elif hit and (hit.get("action") or {}).get("type") == "pause":
                    mode = "gesture"
                else:
                    for act in gest.step(pts, now, click_pts, two=two):
                        kind = act[0]
                        if kind == "move":
                            m = float(cfg.get("margin", 0.15))
                            nx = min(1.0, max(0.0, (act[1] - m) / (1 - 2 * m)))
                            ny = min(1.0, max(0.0, (act[2] - m) / (1 - 2 * m)))
                            # smoothing 0 -> 3.3 Hz (raw-ish), 0.95 -> 0.45 Hz (very steady)
                            fx.mc = fy.mc = 0.3 + (1.0 - min(0.95, max(0.0, float(cfg.get("smoothing", 0.5))))) * 3.0
                            mover.aim(fx(nx, ft) * (sw - 1), fy(ny, ft) * (sh - 1), ft, float(cfg.get("predict", 0.5)))
                            aimed = True
                        elif kind == "touch_down":
                            pt = ctypes.wintypes.POINT()
                            user32.GetCursorPos(ctypes.byref(pt))
                            m = float(cfg.get("margin", 0.15))
                            fh, fw = frame.shape[:2]
                            for f_ in tf:
                                f_.reset()
                            a = (tf[0](act[1][0], ft), tf[1](act[1][1], ft))
                            b = (tf[2](act[2][0], ft), tf[3](act[2][1], ft))
                            touch_anchor.update(cx=pt.x, cy=pt.y, mx=(a[0] + b[0]) / 2, my=(a[1] + b[1]) / 2,
                                                len0=math.hypot((b[0] - a[0]) * fw, (b[1] - a[1]) * fh),
                                                kx=sw / (1 - 2 * m), ky=sh / (1 - 2 * m))
                            clamp = lambda q: (min(sw - 1, max(0, q[0])), min(sh - 1, max(0, q[1])))
                            pts0 = [clamp(q) for q in touch_geometry(a, b, touch_anchor, fw, fh)]
                            if touch.ok and touch.down(pts0):
                                mover.touch_to(pts0)
                                event("Two-hand pinch: spread to resize, twist to rotate, move to pan")
                            else:
                                event("Two-hand pinch: Windows refused touch input on this PC")
                        elif kind == "touch_move":
                            fh, fw = frame.shape[:2]
                            a = (tf[0](act[1][0], ft), tf[1](act[1][1], ft))
                            b = (tf[2](act[2][0], ft), tf[3](act[2][1], ft))
                            pts2 = [(min(sw - 1, max(0, q[0])), min(sh - 1, max(0, q[1])))
                                    for q in touch_geometry(a, b, touch_anchor, fw, fh)]
                            if touch.last is not None:
                                mover.touch_to(pts2)  # the fingers glide there; the mover keeps them alive
                        elif kind == "touch_up":
                            mover.touch_off()
                            touch.up()
                        elif kind == "click":
                            click(LEFTDOWN, LEFTUP)
                            event("Click")
                        elif kind == "down":
                            user32.mouse_event(LEFTDOWN, 0, 0, 0, 0)
                            event("Drag started")
                        elif kind == "up":
                            user32.mouse_event(LEFTUP, 0, 0, 0, 0)
                            event("Drop")
                        elif kind == "right":
                            click(RIGHTDOWN, RIGHTUP)
                            event("Right click")
                        elif kind == "scroll":
                            wheel(act[1])
                        elif kind in ("pause", "resume"):
                            event("Paused (open palm)" if kind == "pause" else "Resumed (open palm)")
                    mode = gest.mode
            if lm is None:
                fx.reset()
                fy.reset()
            if not aimed:
                mover.idle()
            if overlay and game:
                overlay.hide()  # a topmost ring would sit over a fullscreen game
            elif overlay:
                pt = ctypes.wintypes.POINT()
                user32.GetCursorPos(ctypes.byref(pt))
                overlay.show(pt.x, pt.y, mode)
            frames += 1
            if now - fps_t >= 1.0:
                hz, hz_ticks = (mover.ticks - hz_ticks) / (now - fps_t), mover.ticks
                fps, frames, fps_t = frames / (now - fps_t), 0, now
            if now - last_preview > 0.066:
                on_pinch = float(cfg.get("pinch", 0.35)) * 1.4
                click_busy = bool(two and click_pts is not None and (gest.down or gest.rdown or read_hand(click_pts)["pinch"] < on_pinch
                                           or read_hand(click_pts)["rpinch"] < on_pinch))
                if game and rec is None:
                    fh, fw = frame.shape[:2]
                    if stick_p is not None and gm.center is not None:
                        c0 = (int(gm.center[0] * fw), int(gm.center[1] * fh))
                        cv2.circle(frame, c0, int(float(cfg.get("stick_dead", 0.35)) * gm.size * fw), ROLE_BGR["click"], 2, cv2.LINE_AA)
                        cv2.line(frame, c0, (int(stick_p[9][0] * fw), int(stick_p[9][1] * fh)), ROLE_BGR["click"], 3, cv2.LINE_AA)
                    held_txt = " ".join(sorted(game_held)).upper()[:18]
                    for p in seen_hands:
                        is_look = p is look_p
                        draw_hand(cv2, frame, conns, p, ROLE_BGR["pointer"] if is_look else ROLE_BGR["click"],
                                  p is look_p or p is stick_p, ("LOOK" if is_look else "MOVE") + (" " + held_txt if held_txt else ""))
                else:
                    for look in hand_looks(seen_hands, pts, two, mode, (match_info or {}).get("name"),
                                           matcher.last_idx, click_busy, gest.down):
                        draw_hand(cv2, frame, conns, *look)
                h0, w0 = frame.shape[:2]
                small = cv2.resize(frame, (640, 640 * h0 // w0)) if w0 > 640 else frame
                ok2, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok2:
                    write_atomic(P("preview.jpg"), jpg.tobytes(), binary=True)
                last_preview = now
            if now - last_status > 0.2:
                st = {"mode": mode, "fps": round(fps, 1), "hand": lm is not None, "hands": len(lms), "two": two, "t": now,
                      "paused": gest.paused, "match": match_info, "events": list(events), "game": game,
                      "held": sorted(game_held) if game else [],
                      "ms": round(ms, 1), "out_hz": round(hz),  # work per frame, cursor updates per second
                      "warning": no_picture(now - dark_since) if dark_since else ""}
                if rec is not None:
                    st["record"] = {"id": rec["id"], "name": rec["name"], "phase": rec["phase"],
                                    "left": round(max(0.0, rec["until"] - now), 1), "samples": len(rec["samples"])}
                write_atomic(P("status.json"), json.dumps(st))
                last_status = now
            ms += ((time.perf_counter() - ft) * 1000 - ms) * 0.1  # frame arrival -> everything sent, smoothed
    finally:
        mover.close()
        if gest.down:
            user32.mouse_event(LEFTUP, 0, 0, 0, 0)
        touch.up()
        let_go_gesture()
        release_game_keys()
        grab.close()
        if overlay:
            overlay.close()
        try:
            write_atomic(P("status.json"), json.dumps({"mode": "stopped", "t": time.time()}))
        except Exception:
            pass


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    main(args.get("--dir", "."), int(args.get("--parent", "0") or 0))
'''
