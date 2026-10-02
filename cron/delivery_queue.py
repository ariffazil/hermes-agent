"""Profile-local durable handoff for cron delivery through live gateway adapters.

A restart-safe cron worker executes outside the gateway cgroup.  It cannot own
relay/E2EE adapter objects, so it queues the final send here.  A gateway claims
each row at most once.  If that gateway dies after claiming, the outcome is
marked unknown and never retried: losing a delivery is safer than duplicating a
possibly-completed send.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from agent.redact import redact_sensitive_text
from cron.executions import _owner_is_live, _process_start_time
from hermes_constants import get_hermes_home
from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

DELIVERY_DB: Optional[Path] = None
_PROCESS_ID = uuid.uuid4().hex
_lock = threading.RLock()
_ACTIVE_DELIVERIES: set[str] = set()
_TERMINAL = ("delivered", "failed", "unknown", "suppressed")
MAX_TERMINAL_DELIVERIES = 1000
DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS = 300.0


def _prune_terminal_unlocked(conn: sqlite3.Connection) -> None:
    """Redact terminal payloads and retain only bounded outcome metadata.

    2026-10-02 FI-008 delivery-provenance repair (F13 SAH):
    Promoting a delivery to a tombstone now lifts the provenance fields
    (job_id, run_id, source_event_id, audience_class, destination_ref,
    classification, message_fingerprint, queued_at, sent_at) BEFORE the
    payload is scrubbed. After scrub the live row retains outcome-only
    metadata; the tombstone row retains the causal chain. Privacy boundary
    preserved: message body never stored in tombstone; only SHA-256 fingerprint.
    CONTEXT (C-AUDIT-PROVENANCE-2026-10-02)."""
    # 1. Lift provenance into tombstone BEFORE scrub — preserves the
    #    "which job caused this human message?" answer without body bytes.
    conn.execute(
        """INSERT OR IGNORE INTO delivery_tombstones
              (execution_id, terminal_status, finished_at,
               job_id, run_id, source_event_id,
               audience_class, destination_ref, classification,
               message_fingerprint, queued_at, sent_at)
           SELECT execution_id, status, finished_at,
                  job_id, run_id, source_event_id,
                  audience_class, destination_ref, classification,
                  message_fingerprint, created_at,
                  COALESCE(sent_at, created_at)
           FROM deliveries
           WHERE status IN ('delivered','failed','unknown','suppressed')
             AND execution_id NOT IN (SELECT execution_id FROM delivery_tombstones)
             AND (job_id IS NOT NULL OR message_fingerprint IS NOT NULL)"""
    )
    # 2. Scrub the live-row payload. The provenance is now durable in the tombstone.
    conn.execute(
        """UPDATE deliveries SET job_json='{}', content=''
           WHERE status IN ('delivered','failed','unknown','suppressed')
             AND (job_json != '{}' OR content != '')"""
    )
    keep = max(0, int(MAX_TERMINAL_DELIVERIES))
    terminal_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM deliveries "
            "WHERE status IN ('delivered','failed','unknown','suppressed')"
        ).fetchone()[0]
    )
    excess = terminal_count - keep
    if excess > 0:
        conn.execute(
            """INSERT OR IGNORE INTO delivery_tombstones
               (execution_id, terminal_status, finished_at,
                job_id, run_id, source_event_id,
                audience_class, destination_ref, classification,
                message_fingerprint, queued_at, sent_at)
               SELECT execution_id, status, finished_at,
                      job_id, run_id, source_event_id,
                      audience_class, destination_ref, classification,
                      message_fingerprint, created_at,
                      COALESCE(sent_at, created_at)
               FROM deliveries
               WHERE status IN ('delivered','failed','unknown','suppressed')
               ORDER BY finished_at, created_at, execution_id
               LIMIT ?""",
            (excess,),
        )
        conn.execute(
            """DELETE FROM deliveries WHERE execution_id IN (
                 SELECT execution_id FROM deliveries
                 WHERE status IN ('delivered','failed','unknown','suppressed')
                 ORDER BY finished_at, created_at, execution_id
                 LIMIT ?
               )""",
            (excess,),
        )


def queue_path(home: Optional[Path] = None) -> Path:
    """The queue file of ``home`` (the active home when None); a test override wins."""
    if DELIVERY_DB is not None:
        return DELIVERY_DB
    root = Path(home) if home is not None else get_hermes_home()
    return root.resolve() / "cron" / "deliveries.db"


def _path() -> Path:
    return queue_path()


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_cli.sqlite_util import add_column_if_missing

    # SQLite cannot widen a CHECK in place. Preserve all old rows atomically,
    # including claimed sends, while admitting a distinct never-sent disposition.
    for table in ("deliveries", "delivery_tombstones"):
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if row and "'suppressed'" not in row[0]:
            conn.execute("SAVEPOINT notification_disposition")
            try:
                conn.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
                conn.execute(row[0].replace("'unknown'", "'unknown','suppressed'"))
                conn.execute(f"INSERT INTO {table} SELECT * FROM {table}_old")
                conn.execute(f"DROP TABLE {table}_old")
                conn.execute("RELEASE notification_disposition")
            except BaseException:
                conn.execute("ROLLBACK TO notification_disposition")
                conn.execute("RELEASE notification_disposition")
                raise

    conn.execute(
        """CREATE TABLE IF NOT EXISTS deliveries (
             execution_id TEXT PRIMARY KEY,
             job_json TEXT NOT NULL,
             content TEXT NOT NULL,
             for_failure INTEGER NOT NULL DEFAULT 0,
             status TEXT NOT NULL CHECK(status IN
               ('pending','delivering','delivered','failed','unknown','suppressed')),
             owner_process_id TEXT,
             owner_pid INTEGER,
             owner_started_at INTEGER,
             created_at TEXT NOT NULL,
             finished_at TEXT,
             error TEXT
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_tombstones (
             execution_id TEXT PRIMARY KEY,
             terminal_status TEXT NOT NULL CHECK(terminal_status IN
               ('delivered','failed','unknown','suppressed')),
             finished_at TEXT
           )"""
    )
    add_column_if_missing(
        conn, "deliveries", "for_failure",
        "for_failure INTEGER NOT NULL DEFAULT 0",
    )
    # 2026-10-02 FI-008 delivery-provenance repair (F13 SAH).
    # The ``deliveries`` payload is scrubbed (``job_json='{}' content=''``) on terminal
    # to bound retention, but the audit layer still needs to answer: which job caused
    # which human message? Tombstone extended with provenance (no message content).
    # History rows whose job_id was already scrubbed stay UNKNOWN_HISTORICAL — no
    # backfill. CONTEXT (C-AUDIT-PROVENANCE-2026-10-02).
    for col, decl in [
        ("job_id",                  "TEXT"),
        ("run_id",                  "TEXT"),
        ("source_event_id",         "TEXT"),
        ("audience_class",          "TEXT"),
        ("destination_ref",         "TEXT"),
        ("classification",          "TEXT"),
        ("message_fingerprint",     "TEXT"),
        ("queued_at",               "TEXT"),
        ("sent_at",                 "TEXT"),
    ]:
        add_column_if_missing(
            conn, "delivery_tombstones", col, f"{col} {decl}"
        )
    # Same provenance fields on the live deliveries row so the tombstone writer
    # can lift them at terminal time.
    for col, decl in [
        ("job_id",                  "TEXT"),
        ("run_id",                  "TEXT"),
        ("source_event_id",         "TEXT"),
        ("audience_class",          "TEXT"),
        ("destination_ref",         "TEXT"),
        ("classification",          "TEXT"),
        ("message_fingerprint",     "TEXT"),
        ("queued_at",               "TEXT"),
        ("sent_at",                 "TEXT"),
    ]:
        add_column_if_missing(
            conn, "deliveries", col, f"{col} {decl}"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tomb_job_id "
        "ON delivery_tombstones(job_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tomb_audience "
        "ON delivery_tombstones(audience_class)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_deliveries_job_id "
        "ON deliveries(job_id)"
    )


def _connect() -> sqlite3.Connection:
    # Late imports: a scheduler daemon that outlives an on-disk upgrade already has the OLD
    # ``hermes_cli.sqlite_util`` / ``cron.jobs`` cached, so new names must be resolved at call time,
    # not at import time (the guarantee cron/ledger.py used to carry, see e24c8499).
    from hermes_cli.sqlite_util import open_db

    path = _path()
    conn = open_db(path, db_label="cron/deliveries.db", synchronous_full=True, initialize=_initialize_schema)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return conn


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    # Pruning is done explicitly by the paths that create terminal
    # rows (_finish / recover_abandoned / _terminalize_wait_timeout);
    # read-only polls must not pay for a full-table UPDATE + COUNT.
    from hermes_cli.sqlite_util import transaction

    with _lock, transaction(_connect()) as conn:
        yield conn


def enqueue(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
) -> dict:
    """Persist one idempotent delivery request before the worker waits."""
    # Provenance extracted up front so it survives payload scrub.
    prov = _extract_provenance(execution_id, job, content)
    with _transaction() as conn:
        # Serialize the tombstone check and insert with retention in other
        # processes, which can move a terminal delivery into the tombstone table.
        conn.execute("BEGIN IMMEDIATE")
        tombstone = conn.execute(
            "SELECT terminal_status, finished_at FROM delivery_tombstones "
            "WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        if tombstone is not None:
            return {
                "execution_id": str(execution_id),
                "status": tombstone["terminal_status"],
                "finished_at": tombstone["finished_at"],
            }
        conn.execute(
            """INSERT OR IGNORE INTO deliveries
               (execution_id, job_json, content, for_failure, status, created_at)
               VALUES (?, ?, ?, ?, 'pending', ?)""",
            (
                str(execution_id),
                json.dumps(job, ensure_ascii=False, sort_keys=True),
                str(content),
                int(bool(for_failure)),
                _hermes_now().isoformat(),
            ),
        )
        # Persist provenance on the delivery row so the tombstone writer can lift it.
        conn.execute(
            """UPDATE deliveries SET
                  job_id=?, run_id=?, source_event_id=?,
                  audience_class=?, destination_ref=?, classification=?,
                  message_fingerprint=?
               WHERE execution_id=? AND job_id IS NULL""",
            (
                prov.get("job_id"), prov.get("run_id"), prov.get("source_event_id"),
                prov.get("audience_class"), prov.get("destination_ref"), prov.get("classification"),
                prov.get("message_fingerprint"), str(execution_id),
            ),
        )
        row = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (str(execution_id),)
        ).fetchone()
    return dict(row)


def _extract_provenance(execution_id: str, job: dict, content: str) -> dict:
    """Build the durable-provenance payload from job metadata + content fingerprint.

    Returns ONLY non-sensitive fields. Body bytes are never retained — only a SHA-256
    fingerprint over a canonical projection of the outbound message so the audit layer
    can link two receipts of the same message without exposing its body.
    CONTEXT (C-AUDIT-PROVENANCE-2026-10-02)."""
    import hashlib
    job_id = job.get("id") or job.get("job_id")
    deliver = job.get("deliver") or job.get("delivery") or ""
    audience_class = "machine_housekeeping" if str(job.get("lane", "")) == "machine" else "human"
    # Classification is set by the gate script in its stdout block title (CHANGED / WARNING / ERROR / RECOVERED / NO_CHANGE / FALSIFIED).
    classification = None
    head = (content or "").splitlines()[0] if content else ""
    head_up = head.upper()
    for tag in ("ERROR", "WARNING", "RECOVERED", "CHANGED", "FALSIFIED", "NO_CHANGE",
                "DRIFT", "BACKUP", "TRIPWIRE"):
        if tag in head_up:
            classification = tag
            break
    fingerprint_src = (content or "").strip()
    fp = hashlib.sha256(fingerprint_src.encode("utf-8", errors="replace")).hexdigest() if fingerprint_src else None
    return {
        "job_id": job_id,
        "run_id": job_id,           # run_id == job_id at this layer; execution_id is the delivery-instance id
        "source_event_id": job_id,  # source_event_id surfaces the originator (cron schedule); same as job_id today
        "audience_class": audience_class,
        "destination_ref": deliver,
        "classification": classification,
        "message_fingerprint": fp,
    }


def get_status(execution_id: str) -> Optional[dict]:
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (str(execution_id),)
        ).fetchone()
        if row is not None:
            return dict(row)
        tombstone = conn.execute(
            "SELECT execution_id, terminal_status, finished_at "
            "FROM delivery_tombstones WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
    if tombstone is None:
        return None
    return {
        "execution_id": tombstone["execution_id"],
        "status": tombstone["terminal_status"],
        "finished_at": tombstone["finished_at"],
        "error": None,
    }


def claim_next() -> Optional[dict]:
    """Atomically claim one pending send before touching the transport."""
    pid = os.getpid()
    started = _process_start_time(pid)
    with _transaction() as conn:
        row = conn.execute(
            "SELECT execution_id FROM deliveries WHERE status='pending' "
            "ORDER BY created_at, execution_id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        cur = conn.execute(
            """UPDATE deliveries SET status='delivering', owner_process_id=?,
               owner_pid=?, owner_started_at=?
               WHERE execution_id=? AND status='pending'""",
            (_PROCESS_ID, pid, started, row["execution_id"]),
        )
        if cur.rowcount != 1:
            return None
        claimed = conn.execute(
            "SELECT * FROM deliveries WHERE execution_id=?", (row["execution_id"],)
        ).fetchone()
        _ACTIVE_DELIVERIES.add(row["execution_id"])
    result = dict(claimed)
    result["job"] = json.loads(result.pop("job_json"))
    return result


def _finish(execution_id: str, *, error: Optional[str], suppressed: bool = False) -> bool:
    status = "failed" if error else "suppressed" if suppressed else "delivered"
    safe_error = (
        redact_sensitive_text(str(error), force=True, redact_url_credentials=True)
        if error
        else None
    )
    with _transaction() as conn:
        cur = conn.execute(
            """UPDATE deliveries SET status=?, finished_at=?, error=?, sent_at=?
               WHERE execution_id=? AND status='delivering'
                 AND owner_process_id=? AND owner_pid=?""",
            (
                status,
                _hermes_now().isoformat(),
                safe_error,
                _hermes_now().isoformat(),
                execution_id,
                _PROCESS_ID,
                os.getpid(),
            ),
        )
        _prune_terminal_unlocked(conn)
    return cur.rowcount == 1


def recover_abandoned() -> int:
    """Fence dead delivery owners as unknown; never replay uncertain sends."""
    changed = 0
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT execution_id, owner_process_id, owner_pid, owner_started_at "
            "FROM deliveries WHERE status='delivering'"
        ).fetchall()
        for row in rows:
            same_process = row["owner_process_id"] == _PROCESS_ID
            if same_process:
                with _lock:
                    if row["execution_id"] in _ACTIVE_DELIVERIES:
                        continue
            elif _owner_is_live(int(row["owner_pid"]), row["owner_started_at"]):
                continue
            error = (
                "Gateway finished delivery but could not persist its outcome; "
                "send was not retried."
                if same_process
                else "Gateway exited during delivery; send outcome is unknown and was not retried."
            )
            cur = conn.execute(
                """UPDATE deliveries SET status='unknown', finished_at=?, error=?
                   WHERE execution_id=? AND status='delivering'""",
                (
                    _hermes_now().isoformat(),
                    error,
                    row["execution_id"],
                ),
            )
            changed += cur.rowcount
        _prune_terminal_unlocked(conn)
    return changed


def drain(
    send: Callable[[dict, str, bool], Optional[str]], *, limit: int = 20
) -> int:
    """Deliver pending rows through *send*, terminalizing every claimed row."""
    recover_abandoned()
    processed = 0
    for _ in range(max(0, limit)):
        row = claim_next()
        if row is None:
            break
        with _lock:
            _ACTIVE_DELIVERIES.add(row["execution_id"])
        try:
            try:
                error = send(
                    row["job"], row["content"], bool(row["for_failure"])
                )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
            _finish(row["execution_id"], error=error,
                    suppressed=bool(row["job"].get("_notification_all_targets_suppressed")))
        finally:
            with _lock:
                _ACTIVE_DELIVERIES.discard(row["execution_id"])
        processed += 1
    return processed


def _terminalize_wait_timeout(execution_id: str) -> str:
    """Fence a delivery whose worker can no longer wait for confirmation.

    A row still ``pending`` was provably never attempted, so it is left queued
    for whichever gateway comes up next (a restart that includes an update can
    easily exceed the worker's wait budget).  That is a deferral, not a
    failure: report success so the job is not recorded ``delivery_failed`` for
    a message the drain will still send.  Only a row caught mid-send is
    uncertain and gets fenced ``unknown``.
    """
    now = _hermes_now().isoformat()
    uncertain_error = (
        "timed out while gateway delivery was in progress; outcome is unknown and "
        "was not retried"
    )
    with _transaction() as conn:
        row = conn.execute(
            "SELECT status FROM deliveries WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        if row is not None and row["status"] == "pending":
            logger.warning(
                "Cron delivery %s: no live gateway within the wait budget; "
                "left queued for the next gateway",
                execution_id,
            )
            return ""
        conn.execute(
            """UPDATE deliveries SET status='unknown', finished_at=?, error=?
               WHERE execution_id=? AND status='delivering'""",
            (now, uncertain_error, str(execution_id)),
        )
        row = conn.execute(
            "SELECT status, error FROM deliveries WHERE execution_id=?",
            (str(execution_id),),
        ).fetchone()
        _prune_terminal_unlocked(conn)
    if row is None:
        return "timed out waiting for live gateway delivery"
    if row["status"] in {"delivered", "suppressed"}:
        return ""
    return str(row["error"] or f"delivery {row['status']}")


def enqueue_and_wait(
    execution_id: str,
    job: dict,
    content: str,
    *,
    for_failure: bool = False,
    timeout: Optional[float] = None,
) -> Optional[str]:
    """Queue delivery and wait for a gateway's terminal at-most-once outcome."""
    queued = enqueue(execution_id, job, content, for_failure=for_failure)
    if queued["status"] in _TERMINAL:
        return None if queued["status"] in {"delivered", "suppressed"} else str(
            queued.get("error") or f"delivery {queued['status']}"
        )
    wait_timeout = (
        DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS if timeout is None else max(0.0, timeout)
    )
    deadline = time.monotonic() + wait_timeout
    while time.monotonic() < deadline:
        row = get_status(execution_id)
        if row and row["status"] in _TERMINAL:
            return None if row["status"] in {"delivered", "suppressed"} else str(
                row.get("error") or f"delivery {row['status']}"
            )
        time.sleep(1.0)
    return _terminalize_wait_timeout(execution_id) or None
