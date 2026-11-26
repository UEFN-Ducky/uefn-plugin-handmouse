"""On a Mac, Hand Mouse refuses setup and start with a plain reason instead of a ctypes crash."""

import os
import sys

import backend as hm


class FakeApi:
    plugin_id = "handmouse"

    def __init__(self, folder):
        self.folder = str(folder)
        self.data = {}
        self.tools, self.rpcs = {}, {}

    def local_dir(self):
        return self.folder

    def tool(self, **_):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco

    def register_pipeline_node(self, _name):
        return lambda fn: fn

    def register_panel_rpc(self, name, fn):
        self.rpcs[name] = fn


def test_mac_setup_and_start_say_windows_only(tmp_path, monkeypatch):
    api = FakeApi(tmp_path)
    hm.register(api)
    monkeypatch.setattr(sys, "platform", "darwin")
    for name in ("handmouse_setup", "handmouse_start"):
        out = api.tools[name]()
        assert out["ok"] is False and "Windows only" in out["error"]
    assert api.rpcs["running"]() == {"ok": True, "running": False}


def test_mac_pid_check_uses_signal_zero(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert hm._pid_alive(os.getpid()) is True
    assert hm._pid_alive(0) is False
