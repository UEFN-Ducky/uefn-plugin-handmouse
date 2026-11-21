# Hand Mouse

Control the Windows cursor with your hand through the webcam. Point to move, quick pinch to click, hold the pinch to drag, two fingers to scroll. Pinch with both hands to resize, zoom and rotate (real two-finger touch). Game mode: one hand walks with WASD, the other aims the camera, and every control can be re-bound. Record your own gestures for clicks, hotkeys and held game keys. Works in Ducky, UEFN, games and every app. Ctrl+Alt+G toggles game mode.

Code map: `backend/__init__.py` (settings, gestures, MCP tools, panel RPC), `backend/helper_src.py` (the tracking helper: hand landmarks, gestures, cursor / touch / key injection; `selftest()` covers the gesture logic), `ui/` (the panel and header menu), `skills/` (the agent skill).

Desktop plugin for [UEFN-Ducky](https://github.com/UEFN-Ducky/UEFN-Ducky) (`handmouse`).
Install or update from **Settings → Store** in the app. Do not install from a zip by hand.

## License

MIT. Copyright (c) 2026 Mindful Path Company, LLC. See [LICENSE](LICENSE).
