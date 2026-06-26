"""Web console for launching CS2810 Phase 2 matches."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import tyro
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel


PHASE2_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PHASE2_DIR / "phase2_config.yaml"
LOG_DIR = PHASE2_DIR / "logs"
RESULT_DIR = PHASE2_DIR / "results"


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "match"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class MatchRequest(BaseModel):
    shooter_team: str
    shooter_api: str
    goalkeeper_team: str
    goalkeeper_api: str
    num_trials: int = 10
    match_id: str | None = None


@dataclass
class Slot:
    name: str
    port: int
    external_url: str
    match_id: str | None = None


@dataclass
class Match:
    match_id: str
    shooter_team: str
    shooter_api: str
    goalkeeper_team: str
    goalkeeper_api: str
    num_trials: int
    status: str = "queued"
    created_at: str = field(default_factory=_timestamp)
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    slot_name: str | None = None
    viser_url: str | None = None
    log_path: str | None = None
    result_path: str | None = None
    returncode: int | None = None
    summary: dict[str, Any] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "shooter_team": self.shooter_team,
            "shooter_api": self.shooter_api,
            "goalkeeper_team": self.goalkeeper_team,
            "goalkeeper_api": self.goalkeeper_api,
            "num_trials": self.num_trials,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "pid": self.pid,
            "slot_name": self.slot_name,
            "viser_url": self.viser_url,
            "log_path": self.log_path,
            "result_path": self.result_path,
            "returncode": self.returncode,
            "summary": self.summary,
            "error": self.error,
        }


class MatchManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.matches: dict[str, Match] = {}
        self.queue: list[str] = []
        self.processes: dict[str, subprocess.Popen[Any]] = {}
        self.slots = [
            Slot(
                name=str(slot["name"]),
                port=int(slot["port"]),
                external_url=str(slot["external_url"]),
            )
            for slot in config["viser_slots"]
        ]
        self.lock = asyncio.Lock()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)

    def _make_match_id(self, req: MatchRequest) -> str:
        if req.match_id:
            base = req.match_id
        else:
            base = f"{req.shooter_team}_shooter_vs_{req.goalkeeper_team}_goalkeeper"
        candidate = f"{_timestamp()}_{_sanitize(base)}"
        suffix = 2
        unique = candidate
        while unique in self.matches:
            unique = f"{candidate}_{suffix}"
            suffix += 1
        return unique

    def _free_slot(self) -> Slot | None:
        for slot in self.slots:
            if slot.match_id is None:
                return slot
        return None

    async def create_match(self, req: MatchRequest) -> Match:
        if req.num_trials <= 0:
            raise HTTPException(status_code=400, detail="num_trials must be positive")
        async with self.lock:
            match_id = self._make_match_id(req)
            match = Match(
                match_id=match_id,
                shooter_team=req.shooter_team,
                shooter_api=req.shooter_api,
                goalkeeper_team=req.goalkeeper_team,
                goalkeeper_api=req.goalkeeper_api,
                num_trials=req.num_trials,
            )
            self.matches[match_id] = match
            self.queue.append(match_id)
            await self._schedule_locked()
            return match

    async def stop_match(self, match_id: str) -> Match:
        async with self.lock:
            match = self._get_match(match_id)
            if match.status == "queued":
                self.queue = [mid for mid in self.queue if mid != match_id]
                match.status = "stopped"
                match.finished_at = _timestamp()
                return match
            proc = self.processes.get(match_id)
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
                match.status = "stopped"
                match.finished_at = _timestamp()
            return match

    def _get_match(self, match_id: str) -> Match:
        try:
            return self.matches[match_id]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="match not found") from exc

    async def _schedule_locked(self) -> None:
        while self.queue:
            slot = self._free_slot()
            if slot is None:
                return
            match_id = self.queue.pop(0)
            match = self.matches[match_id]
            if match.status != "queued":
                continue
            self._start_process(match, slot)

    def _start_process(self, match: Match, slot: Slot) -> None:
        slot.match_id = match.match_id
        match.status = "running"
        match.started_at = _timestamp()
        match.slot_name = slot.name
        match.viser_url = slot.external_url

        log_path = LOG_DIR / f"{match.match_id}.log"
        result_path = RESULT_DIR / f"{match.match_id}.json"
        match.log_path = str(log_path)
        match.result_path = str(result_path)

        cmd = [
            sys.executable,
            str(PHASE2_DIR / "compete.py"),
            "--shooter-api",
            match.shooter_api,
            "--goalkeeper-api",
            match.goalkeeper_api,
            "--shooter-team",
            match.shooter_team,
            "--goalkeeper-team",
            match.goalkeeper_team,
            "--match-id",
            match.match_id,
            "--num-trials",
            str(match.num_trials),
            "--config-path",
            str(CONFIG_PATH),
            "--results-json",
            str(result_path),
            "--viser-host",
            "0.0.0.0",
            "--viser-port",
            str(slot.port),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HOME", "/tmp")
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PHASE2_DIR.parent),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        match.pid = proc.pid
        self.processes[match.match_id] = proc

    async def monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            async with self.lock:
                for match_id, proc in list(self.processes.items()):
                    rc = proc.poll()
                    if rc is None:
                        continue
                    match = self.matches[match_id]
                    match.returncode = rc
                    match.finished_at = match.finished_at or _timestamp()
                    if match.status not in {"stopped"}:
                        match.status = "done" if rc == 0 else "failed"
                    self._load_summary(match)
                    self.processes.pop(match_id, None)
                    for slot in self.slots:
                        if slot.match_id == match_id:
                            slot.match_id = None
                            break
                await self._schedule_locked()

    def _load_summary(self, match: Match) -> None:
        if not match.result_path:
            return
        path = Path(match.result_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            match.summary = data.get("summary")
            match.error = data.get("fatal_error")
        except Exception as exc:
            match.error = f"failed to read result json: {exc}"

    def list_matches(self) -> list[dict[str, Any]]:
        return [m.as_dict() for m in sorted(self.matches.values(), key=lambda x: x.created_at, reverse=True)]

    def get_logs(self, match_id: str, max_lines: int = 200) -> str:
        match = self._get_match(match_id)
        if not match.log_path:
            return ""
        path = Path(match.log_path)
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])

    def get_result(self, match_id: str) -> dict[str, Any]:
        match = self._get_match(match_id)
        if not match.result_path or not Path(match.result_path).exists():
            raise HTTPException(status_code=404, detail="result not ready")
        return json.loads(Path(match.result_path).read_text(encoding="utf-8"))


def create_app(config: dict[str, Any]) -> FastAPI:
    manager = MatchManager(config)
    app = FastAPI(title="CS2810 Phase 2 Tournament Console")
    app.state.manager = manager

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.monitor_task = asyncio.create_task(manager.monitor_loop())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        app.state.monitor_task.cancel()
        for proc in manager.processes.values():
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _INDEX_HTML.replace("__WEB_EXTERNAL_URL__", config["web"]["external_url"])

    @app.post("/api/matches")
    async def create_match(req: MatchRequest) -> dict[str, Any]:
        match = await manager.create_match(req)
        return match.as_dict()

    @app.get("/api/matches")
    async def list_matches() -> list[dict[str, Any]]:
        return manager.list_matches()

    @app.get("/api/matches/{match_id}")
    async def get_match(match_id: str) -> dict[str, Any]:
        return manager._get_match(match_id).as_dict()

    @app.post("/api/matches/{match_id}/stop")
    async def stop_match(match_id: str) -> dict[str, Any]:
        match = await manager.stop_match(match_id)
        return match.as_dict()

    @app.get("/api/matches/{match_id}/logs", response_class=PlainTextResponse)
    async def logs(match_id: str) -> str:
        return manager.get_logs(match_id)

    @app.get("/api/matches/{match_id}/result")
    async def result(match_id: str) -> dict[str, Any]:
        return manager.get_result(match_id)

    return app


_INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CS2810 Phase 2</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #20242a; }
    header { background: #1f2937; color: white; padding: 16px 24px; }
    main { max-width: 1180px; margin: 0 auto; padding: 20px; }
    section { background: white; border: 1px solid #d9dee7; border-radius: 6px; padding: 16px; margin-bottom: 18px; }
    h1 { font-size: 22px; margin: 0 0 4px; }
    h2 { font-size: 17px; margin: 0 0 12px; }
    label { display: block; font-size: 13px; font-weight: 700; margin-bottom: 4px; }
    input { width: 100%; box-sizing: border-box; padding: 9px; border: 1px solid #c8ced8; border-radius: 4px; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .row { display: grid; grid-template-columns: 1fr 1fr 120px; gap: 12px; align-items: end; }
    button { background: #2563eb; color: white; border: 0; border-radius: 4px; padding: 10px 12px; cursor: pointer; }
    button.secondary { background: #4b5563; }
    button.danger { background: #b91c1c; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; }
    th { background: #f3f4f6; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    pre { white-space: pre-wrap; background: #101827; color: #d1d5db; padding: 12px; border-radius: 4px; min-height: 120px; }
    .muted { color: #6b7280; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e5e7eb; }
    .running { background: #dbeafe; }
    .done { background: #dcfce7; }
    .failed, .stopped { background: #fee2e2; }
  </style>
</head>
<body>
  <header>
    <h1>CS2810 Phase 2 Tournament Console</h1>
    <div class="muted">External console URL: __WEB_EXTERNAL_URL__</div>
  </header>
  <main>
    <section>
      <h2>Start Match</h2>
      <form id="match-form">
        <div class="grid">
          <div><label>Shooter Team</label><input id="shooter_team" required></div>
          <div><label>Shooter API URL</label><input id="shooter_api" required placeholder="http://host:port"></div>
          <div><label>Goalkeeper Team</label><input id="goalkeeper_team" required></div>
          <div><label>Goalkeeper API URL</label><input id="goalkeeper_api" required placeholder="http://host:port"></div>
        </div>
        <div class="row" style="margin-top:12px;">
          <div><label>Match ID (optional)</label><input id="match_id"></div>
          <div><label>Trials</label><input id="num_trials" type="number" min="1" value="10"></div>
          <button type="submit">Start Match</button>
        </div>
      </form>
    </section>
    <section>
      <h2>Matches</h2>
      <table>
        <thead><tr><th>Status</th><th>Match</th><th>APIs</th><th>Viser</th><th>Summary</th><th>Actions</th></tr></thead>
        <tbody id="matches"></tbody>
      </table>
    </section>
    <section>
      <h2>Log Tail</h2>
      <div class="muted" id="selected">Select a match to view logs.</div>
      <pre id="logs"></pre>
    </section>
  </main>
<script>
let selectedMatch = null;

document.getElementById("match-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const body = {
    shooter_team: document.getElementById("shooter_team").value,
    shooter_api: document.getElementById("shooter_api").value,
    goalkeeper_team: document.getElementById("goalkeeper_team").value,
    goalkeeper_api: document.getElementById("goalkeeper_api").value,
    num_trials: Number(document.getElementById("num_trials").value || 10),
    match_id: document.getElementById("match_id").value || null
  };
  const resp = await fetch("/api/matches", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  if (!resp.ok) alert(await resp.text());
  await refresh();
});

async function refresh() {
  const resp = await fetch("/api/matches");
  const matches = await resp.json();
  const tbody = document.getElementById("matches");
  tbody.innerHTML = "";
  for (const m of matches) {
    const tr = document.createElement("tr");
    const summary = m.summary ? `goals=${m.summary.goals}, gk=${m.summary.goalkeeper_wins}, winner=${m.summary.winner_decision}` : "";
    tr.innerHTML = `
      <td><span class="pill ${m.status}">${m.status}</span></td>
      <td><code>${m.match_id}</code><br>${m.shooter_team} shooter vs ${m.goalkeeper_team} goalkeeper</td>
      <td><code>${m.shooter_api}</code><br><code>${m.goalkeeper_api}</code></td>
      <td>${m.viser_url ? `<a href="${m.viser_url}" target="_blank">${m.viser_url}</a>` : ""}</td>
      <td>${summary}</td>
      <td>
        <button class="secondary" onclick="selectMatch('${m.match_id}')">Logs</button>
        ${m.result_path ? `<a href="/api/matches/${m.match_id}/result" target="_blank"><button class="secondary">JSON</button></a>` : ""}
        ${(m.status === "running" || m.status === "queued") ? `<button class="danger" onclick="stopMatch('${m.match_id}')">Stop</button>` : ""}
      </td>`;
    tbody.appendChild(tr);
  }
  if (selectedMatch) await loadLogs(selectedMatch);
}

async function selectMatch(matchId) {
  selectedMatch = matchId;
  document.getElementById("selected").textContent = matchId;
  await loadLogs(matchId);
}

async function loadLogs(matchId) {
  const resp = await fetch(`/api/matches/${matchId}/logs`);
  document.getElementById("logs").textContent = await resp.text();
}

async function stopMatch(matchId) {
  await fetch(`/api/matches/${matchId}/stop`, {method: "POST"});
  await refresh();
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


@dataclass
class ServerConfig:
    host: str | None = None
    port: int | None = None


def main() -> None:
    config = _load_config()
    args = tyro.cli(ServerConfig, prog="phase2-tournament-server")
    host = args.host or config["web"]["host"]
    port = args.port or int(config["web"]["port"])
    app = create_app(config)
    print(f"[INFO] Tournament console: {config['web']['external_url']}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
