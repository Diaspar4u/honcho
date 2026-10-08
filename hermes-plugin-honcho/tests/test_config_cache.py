"""External-plugin transport, object-generation and explicit observation regressions."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from hermes_honcho import client, client_cache, session


@pytest.fixture(autouse=True)
def clear_clients():
    client.reset_honcho_client()
    yield
    client.close_honcho_clients()
    client.reset_honcho_client()


def test_bound_client_rebuilds_for_yaml_endpoint(tmp_path, monkeypatch):
    from hermes_cli import config

    settings = {"honcho": {"base_url": "https://first.example/v3"}}
    monkeypatch.setattr(config, "load_config", lambda: settings)
    monkeypatch.setattr(config, "load_config_readonly", lambda: settings)
    cfg = client.HonchoClientConfig(
        api_key="test-only", config_path=tmp_path / "honcho.json", hermes_home=tmp_path
    )
    before = client.get_honcho_client(cfg)
    settings["honcho"]["base_url"] = "https://second.example/v3/"
    after = client.get_honcho_client(cfg)
    assert after is not before
    assert "second.example" in str(after._http.base_url)
    assert client_cache._client_cache_key(cfg)[3] == "https://second.example"


@pytest.mark.parametrize("change", [{"workspace": "other"}, {"baseUrl": "https://other.example"}])
def test_ambient_client_rebuilds_for_effective_destination(tmp_path, monkeypatch, change):
    path = tmp_path / "honcho.json"
    values = {"apiKey": "test-only", "workspace": "first", "baseUrl": "https://first.example"}
    monkeypatch.setattr(client, "resolve_config_path", lambda: path)
    path.write_text(json.dumps(values))
    before = client.get_honcho_client()
    values.update(change)
    path.write_text(json.dumps(values))
    assert client.get_honcho_client() is not before


def test_equivalent_endpoint_spelling_keeps_client(tmp_path):
    cfg = client.HonchoClientConfig(
        api_key="test-only", base_url="https://same.example/v3/",
        config_path=tmp_path / "honcho.json", hermes_home=tmp_path,
    )
    before = client.get_honcho_client(cfg)
    cfg.base_url = "https://same.example"
    assert client.get_honcho_client(cfg) is before


def test_manager_clears_sdk_objects_when_client_changes(monkeypatch):
    old, new = object(), object()
    current = [old]
    monkeypatch.setattr(session, "get_honcho_client", lambda cfg: current[0])
    manager = session.HonchoSessionManager(honcho=old)
    manager._peers_cache["user"] = object()
    manager._sessions_cache["conversation"] = object()
    current[0] = new
    assert manager.honcho is new
    assert not manager._peers_cache and not manager._sessions_cache
    assert manager._client_generation == 1
    assert manager.honcho is new
    assert manager._client_generation == 1


def test_inflight_sdk_resolution_retries_across_client_change(monkeypatch):
    old, new = object(), object()
    current = [old]
    monkeypatch.setattr(session, "get_honcho_client", lambda cfg: current[0])
    manager = session.HonchoSessionManager(honcho=old)
    started, release = threading.Event(), threading.Event()
    calls = []

    def fetch():
        value = current[0]
        calls.append(value)
        if value is old:
            started.set()
            assert release.wait(5)
        return value

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(manager._cached_sdk_object, manager._peers_cache, "user", fetch)
        try:
            assert started.wait(5)
            current[0] = new
            assert manager.honcho is new
        finally:
            release.set()
        assert future.result(timeout=5) is new
    assert calls == [old, new]
    assert manager._peers_cache["user"] is new


@pytest.mark.parametrize("local", [{"observation": {}}, {"observationMode": "directional"}])
def test_explicit_observation_is_recorded_from_local_config(tmp_path, monkeypatch, local):
    path = tmp_path / "honcho.json"
    path.write_text(json.dumps({"apiKey": "test-only", **local}))
    monkeypatch.setattr(client, "resolve_config_path", lambda: path)
    assert client.HonchoClientConfig.from_global_config().observation_explicit is True
    path.write_text(json.dumps({"apiKey": "test-only"}))
    assert client.HonchoClientConfig.from_global_config().observation_explicit is False


@pytest.mark.parametrize("explicit", [True, False])
def test_observation_updates_only_explicit_local_policy(monkeypatch, explicit):
    cfg = client.HonchoClientConfig(
        user_observe_me=True, user_observe_others=True,
        ai_observe_me=True, ai_observe_others=False,
    )
    cfg.observation_explicit = explicit
    manager = session.HonchoSessionManager(config=cfg)
    server = {
        "user": SimpleNamespace(observe_me=True, observe_others=False),
        "assistant": SimpleNamespace(observe_me=True, observe_others=True),
    }
    updates = []

    def update(peer, flags):
        updates.append((peer, flags.observe_others))
        server[peer] = flags

    sdk_session = SimpleNamespace(
        add_peers=lambda peers: None,
        get_peer_configuration=lambda peer: server[peer],
        set_peer_configuration=update,
    )
    monkeypatch.setattr(manager, "_sdk_session", lambda key: sdk_session)
    monkeypatch.setattr(manager, "_authed_call", lambda op, call: call())
    flags = manager._configure_session_peers("conversation", "user", "assistant")
    if explicit:
        assert updates == [("assistant", False), ("user", True)]
        assert flags["user_observe_others"] is True
        assert flags["ai_observe_others"] is False
        assert manager._configure_session_peers("conversation", "user", "assistant") == flags
        assert len(updates) == 2
    else:
        assert updates == []
        assert flags["user_observe_others"] is False
        assert flags["ai_observe_others"] is True
