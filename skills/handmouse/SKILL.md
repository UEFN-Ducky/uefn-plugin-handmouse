---
name: handmouse
description: "Hand Mouse: control the Windows cursor with webcam hand gestures (move, click, drag, right click, scroll) and record custom hand poses bound to clicks or hotkeys."
---

# Hand Mouse

Webcam hand tracking (MediaPipe) that drives the real Windows cursor in Ducky, UEFN and any app.

- `handmouse_status()`: setup progress (`setup_progress.steps`, percent, log), running, mode (`move`, `drag`, `scroll`, `gesture`, `recording`, `paused`, `nohand`), matched custom gesture, recent `events`, `last_record`.
- `handmouse_setup()`: one-time install (Python 3.9-3.12 venv + mediapipe + opencv, ~100 MB) in the background; poll `handmouse_status` for progress until `setup` is `ready`.
- `handmouse_start()` / `handmouse_stop()`: start or stop tracking. Start takes over the cursor: only on the person's request.
- `handmouse_settings()` / `handmouse_save_settings(fields)`: `camera`, `smoothing`, `margin`, `pinch`, `scroll_speed`, `overlay`, `palm_hold`, `builtins` ({move, click, right, scroll, palm}: true/false).
- `handmouse_gestures()`: custom gestures (no samples).
- `handmouse_record_gesture(name, action, seconds?, gesture_id?)`: record a static hand pose from the webcam (starts tracking if needed; 3 s countdown then `seconds` of capture). `gesture_id` re-records. Actions: `{type: click|double_click|right_click|middle_click|scroll|keys|pause|none, keys: "ctrl+tab", amount: 3}`.
- `handmouse_save_gesture(gesture_id, fields)`: name, action, tolerance (0.03-0.5, higher = looser), hold (s), enabled.
- `handmouse_delete_gesture(gesture_id)`.
- `handmouse_preview()`: latest camera frame as a JPEG data URI.
- Workflow nodes: `handmouse.start` (needs `confirm`), `handmouse.stop`, `handmouse.status`, `handmouse.gestures`.

Built-in gestures: point = move; thumb+index pinch = click, hold = drag; thumb+middle pinch = right click; index+middle up and moving = scroll; open palm 1 s = pause/resume. A matched custom pose takes priority over built-ins.

The UI design brief for redesigning the panel is `docs/UI_BRIEF.md` in the plugin.
