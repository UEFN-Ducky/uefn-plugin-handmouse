---
name: handmouse
description: "Hand Mouse: control the Windows cursor and games with webcam hand gestures (move, click, drag, right click, scroll, two-hand touch, game mode with re-bindable keys and per-game profiles), tune its speed and feel, and record custom hand poses bound to clicks or hotkeys."
---

# Hand Mouse

Webcam hand tracking (MediaPipe) that drives the real Windows cursor, keyboard and touch input in Ducky, UEFN, games and any app.

- `handmouse_status()`: setup progress (`setup_progress.steps`, percent, log), running, mode (`move`, `drag`, `scroll`, `touch`, `game`, `gesture`, `recording`, `paused`, `nohand`), matched custom gesture, `held` game keys, recent `events`, `last_record`, and speed: `fps` (camera frames a second), `ms` (frame arrival to input sent), `out_hz` (cursor updates a second).
- `handmouse_setup()`: one-time install (Python 3.9-3.12 venv + mediapipe + opencv, ~100 MB) in the background; poll `handmouse_status` for progress until `setup` is `ready`.
- `handmouse_start()` / `handmouse_stop()`: start or stop tracking. Start takes over the cursor: only on the person's request.
- `handmouse_settings()`: every setting as `fields[name] = {value, default, min, max, help}`. Read it before changing anything.
- `handmouse_save_settings(fields)`: change any of them, e.g. `{"smoothing": 0.7}`, `{"game_mode": true}`, `{"game_keys": {"pinch": "e", "look_pinch": "lmb"}}`, `{"builtins": {"scroll": false}}`. Invalid keys keep the old value.
- `handmouse_profile(action, name, fields?)`: per-game control presets. `save` stores the current game keys and feel under `name` (plus `fields` overrides), `apply` switches to them, `list`, `delete`.
- `handmouse_cameras(refresh?)`, `handmouse_reset_phone_camera()`: pick or fix the camera.
- `handmouse_gestures()`: custom gestures (no samples).
- `handmouse_record_gesture(name, action, seconds?, gesture_id?)`: record a static hand pose from the webcam (starts tracking if needed; 3 s countdown then `seconds` of capture). `gesture_id` re-records. Actions: `{type: click|double_click|right_click|middle_click|scroll|keys|pause|none, keys: "ctrl+tab", amount: 3, hold: true}` (`hold`: keep the keys down while the pose lasts). `handmouse_cancel_record()` stops it.
- `handmouse_save_gesture(gesture_id, fields)`: name, action, tolerance (0.03-0.5, higher = looser), hold (s), enabled.
- `handmouse_delete_gesture(gesture_id)`.
- `handmouse_preview()`: latest camera frame as a JPEG data URI.
- Workflow nodes: `handmouse.start` (needs `confirm`), `handmouse.stop`, `handmouse.status`, `handmouse.gestures`.

Built-in gestures: move the hand = move; quick thumb+index pinch = click, held pinch = drag; thumb+middle pinch = right click; index+middle up and moving = scroll; both hands pinched = two-finger touch (resize, zoom, rotate, pan); open palm 1 s = pause/resume. A matched custom pose takes priority over built-ins.

Game mode (`game_mode`, Ctrl+Alt+G): the stick hand is a thumbstick that holds direction keys, the look hand turns the camera with relative mouse movement, pinches on either hand are buttons. Set a game up: `handmouse_save_settings({"game_keys": {...}, "look_speed": 5})`, then `handmouse_profile("save", "<game>")`.

Tuning the feel: cursor shakes at rest → raise `smoothing`; cursor overshoots when the hand stops → lower `predict`; feels behind the hand → raise `predict`, lower `smoothing`; clicks miss → raise `pinch`; walking starts too easily in game mode → raise `stick_dead`. Low `fps` (under 25) means the camera or the light is the limit, not the settings.
