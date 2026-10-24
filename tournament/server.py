import argparse
import base64
import hashlib
import importlib.util
import logging
import os
import secrets
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request
from google.protobuf.message import DecodeError
from pyrusgeom.geom_2d import Vector2D


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for path in (PROJECT_ROOT / "support" / "runtime", PROJECT_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import service_pb2 as pb2
from src.student.framework.runtime import arm
from src.student.framework.world import AgentView
from src.utils.memory import Memory


PROTOCOL_VERSION = 1
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
sessions = {}
sessions_lock = threading.Lock()
request_count = 0
request_count_lock = threading.Lock()
agent_path = None
agent_class_name = "Agent1"


class _Strategy:
    def __init__(self, home_position):
        self.home_position = home_position

    def get_position(self, uniform_number, agent):
        if self.home_position is None:
            return None
        return Vector2D(*self.home_position)


class _Budget:
    __slots__ = ("_match_ms", "_spent_ms", "_cpu_started", "_deadline")

    def __init__(self, match_ms, spent_ms, cpu_started, deadline):
        object.__setattr__(self, "_match_ms", match_ms)
        object.__setattr__(self, "_spent_ms", spent_ms)
        object.__setattr__(self, "_cpu_started", cpu_started)
        object.__setattr__(self, "_deadline", deadline)

    @property
    def remaining_ms(self):
        if self._match_ms <= 0:
            return float("inf")
        elapsed = max(0.0, (time.thread_time() - self._cpu_started) * 1000.0)
        return max(0.0, self._match_ms - self._spent_ms - elapsed)

    @property
    def decision_remaining_ms(self):
        return max(0.0, (self._deadline - time.monotonic()) * 1000.0)


class _Agent:
    def __init__(self, server_params, player_params, player_types, memory,
                 home_position):
        self.wm = None
        self.server_params = server_params
        self.player_params = player_params
        self.player_types = player_types
        self.memory = Memory()
        self.memory.dash_distance_tables.update(memory.get("dash_distance_tables", {}))
        self.strategy = _Strategy(home_position)
        self.actions = []
        self.budget = None
        self.logger = logging.getLogger("tournament.student")

    def get_player_type(self, type_id):
        return self.player_types[max(0, int(type_id))]

    def add_action(self, action):
        self.actions.append(action)

    def add_log_message(self, *args, **kwargs):
        return None


class _Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.student = None
        self.raw_agent = None
        self.module_name = None
        self.match_ms = 0.0
        self.ceiling_ms = 1000.0
        self.hard_ms = 1500.0
        self.max_cuts = 5
        self.spent_ms = 0.0
        self.cuts = 0
        self.benched = False
        self.touched = time.monotonic()


def _authorized():
    expected = "Bearer " + app.config.get("API_TOKEN", "")
    supplied = request.headers.get("Authorization", "")
    return bool(app.config.get("API_TOKEN")) and secrets.compare_digest(supplied, expected)


def _decode(value, message_type):
    try:
        raw = base64.b64decode(value, validate=True)
        message = message_type()
        message.ParseFromString(raw)
        return message
    except (TypeError, ValueError, DecodeError) as exc:
        raise ValueError("invalid protobuf field") from exc


def _load_student_module(session, session_key):
    module_name = "tournament_submission_" + hashlib.sha256(
        session_key.encode("utf-8")).hexdigest()[:24]
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the configured agent file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        student_class = getattr(module, agent_class_name)
        student = student_class()
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not callable(getattr(student, "on_ball", None)) or not callable(
        getattr(student, "off_ball", None)
    ):
        sys.modules.pop(module_name, None)
        raise RuntimeError("the configured class must define on_ball and off_ball")
    session.student = student
    session.module_name = module_name
    return module


def _initialize(session, payload, session_key):
    init = payload.get("init")
    if not isinstance(init, dict):
        return False
    started_cpu = time.thread_time()
    server_params = _decode(init.get("server_params"), pb2.ServerParam)
    player_params = _decode(init.get("player_params"), pb2.PlayerParam)
    player_types = {}
    for key, encoded in init.get("player_types", {}).items():
        player_type = _decode(encoded, pb2.PlayerType)
        player_types[int(key)] = player_type
    memory = init.get("memory", {})
    tables = memory.get("dash_distance_tables", {})
    memory = {
        "dash_distance_tables": {
            int(key): [float(value) for value in values]
            for key, values in tables.items()
        }
    }
    budget = payload.get("budget") or {}
    session.match_ms = float(budget.get("match_ms", 0.0))
    session.ceiling_ms = max(1.0, float(budget.get("ceiling_ms", 1000.0)))
    session.hard_ms = max(session.ceiling_ms, float(budget.get("hard_ms", 1500.0)))
    session.max_cuts = max(0, int(budget.get("max_cuts", 5)))
    _load_student_module(session, session_key)
    session.raw_agent = _Agent(
        server_params,
        player_params,
        player_types,
        memory,
        payload.get("home_position"),
    )
    session.spent_ms += max(0.0, (time.thread_time() - started_cpu) * 1000.0)
    return True


def _get_session(match_id, player_id):
    key = f"{match_id}:{player_id}"
    with sessions_lock:
        session = sessions.get(key)
        if session is None:
            session = _Session()
            sessions[key] = session
    return key, session


def _cleanup_old_sessions():
    now = time.monotonic()
    with sessions_lock:
        expired = [key for key, session in sessions.items()
                   if now - session.touched > 6 * 60 * 60
                   and session.lock.acquire(blocking=False)]
        for key in expired:
            session = sessions.pop(key)
            if session.module_name:
                sys.modules.pop(session.module_name, None)
            session.lock.release()


@app.get("/health")
def health():
    return jsonify(status="ok", protocol_version=PROTOCOL_VERSION)


@app.post("/v1/check")
def check():
    if not _authorized():
        return jsonify(error="unauthorized"), 401
    return jsonify(status="ok", protocol_version=PROTOCOL_VERSION)


@app.post("/v1/act")
def act():
    global request_count
    if not _authorized():
        return jsonify(error="unauthorized"), 401
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.get("protocol_version") != PROTOCOL_VERSION:
        return jsonify(error="unsupported request"), 400
    phase = payload.get("phase")
    match_id = str(payload.get("match_id", ""))
    player_id = str(payload.get("player_id", ""))
    if phase not in ("on_ball", "off_ball") or not match_id or not player_id:
        return jsonify(error="invalid request"), 400

    with request_count_lock:
        request_count += 1
        cleanup = request_count % 200 == 0
    if cleanup:
        _cleanup_old_sessions()

    key, session = _get_session(match_id, player_id)
    try:
        with session.lock:
            session.touched = time.monotonic()
            if session.student is None and not _initialize(session, payload, key):
                return jsonify(error="session initialization required"), 409
            world = _decode(payload.get("world_model"), pb2.WorldModel)
            session.raw_agent.wm = world
            session.raw_agent.strategy.home_position = payload.get("home_position")
            session.raw_agent.actions.clear()

            budget = payload.get("budget") or {}
            session.match_ms = float(budget.get("match_ms", session.match_ms))
            session.ceiling_ms = max(1.0, float(budget.get("ceiling_ms", session.ceiling_ms)))
            session.hard_ms = max(session.ceiling_ms, float(budget.get("hard_ms", session.hard_ms)))
            session.max_cuts = max(0, int(budget.get("max_cuts", session.max_cuts)))
            remaining_before = (
                max(0.0, session.match_ms - session.spent_ms)
                if session.match_ms > 0 else float("inf")
            )
            if session.benched or remaining_before <= 0:
                return jsonify(actions=[], cuts=session.cuts, benched=session.benched)

            start_wall = time.monotonic()
            start_cpu = time.thread_time()
            soft_ms = min(session.ceiling_ms, remaining_before)
            deadline = start_wall + soft_ms / 1000.0
            session.raw_agent.budget = _Budget(
                session.match_ms, session.spent_ms, start_cpu, deadline
            )
            view = AgentView(session.raw_agent)
            getattr(session.student, phase)(view)
            cpu_ms = max(0.0, (time.thread_time() - start_cpu) * 1000.0)
            wall_ms = max(0.0, (time.monotonic() - start_wall) * 1000.0)
            session.spent_ms += cpu_ms
            timed_out = wall_ms >= soft_ms
            if timed_out:
                session.cuts += 1
                if session.max_cuts and session.cuts >= session.max_cuts:
                    session.benched = True
            actions = [] if timed_out else [
                base64.b64encode(action.SerializeToString()).decode("ascii")
                for action in session.raw_agent.actions
            ]
            return jsonify(
                actions=actions,
                cpu_ms=cpu_ms,
                wall_ms=wall_ms,
                remaining_ms=(
                    max(0.0, session.match_ms - session.spent_ms)
                    if session.match_ms > 0 else None
                ),
                timed_out=timed_out,
                cuts=session.cuts,
                benched=session.benched,
            )
    except Exception:
        app.logger.exception("tournament decision failed")
        return jsonify(error="agent decision failed"), 500


@app.post("/v1/end_match")
def end_match():
    if not _authorized():
        return jsonify(error="unauthorized"), 401
    payload = request.get_json(silent=True) or {}
    match_id = str(payload.get("match_id", ""))
    if not match_id:
        return jsonify(error="invalid request"), 400
    prefix = match_id + ":"
    with sessions_lock:
        keys = [key for key in sessions if key.startswith(prefix)]
        for key in keys:
            session = sessions[key]
            if not session.lock.acquire(blocking=False):
                continue
            sessions.pop(key, None)
            if session.module_name:
                sys.modules.pop(session.module_name, None)
            session.lock.release()
    return jsonify(status="ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="src/student/agent.py")
    parser.add_argument("--class", dest="agent_class", default="Agent1")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--token")
    args = parser.parse_args()

    global agent_path, agent_class_name
    agent_path = Path(args.agent).expanduser()
    if not agent_path.is_absolute():
        agent_path = PROJECT_ROOT / agent_path
    agent_path = agent_path.resolve()
    if not agent_path.is_file():
        parser.error(f"agent file not found: {agent_path}")
    agent_class_name = args.agent_class
    os.environ["SANDBOX"] = "1"
    arm([str(agent_path)])
    app.config["API_TOKEN"] = args.token or secrets.token_urlsafe(32)
    print(f"Tournament token: {app.config['API_TOKEN']}", flush=True)
    print(f"Agent: {agent_path} ({agent_class_name})", flush=True)
    print(f"Listening on {args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
