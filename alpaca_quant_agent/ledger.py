"""SQLite-backed ledger: trades taken, gate/skip decisions, Claude's
rationale text, IV history samples (for iv_rank), and daily equity snapshots.
This is the system of record the WRITEUP.md performance section and any
post-hoc review of "how did the agent decide X" is built from.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT,
    position_group TEXT,               -- ties an 'open' row to its later 'close' row
    symbol TEXT NOT NULL,
    sleeve TEXT NOT NULL,
    strategy_type TEXT NOT NULL,
    action TEXT NOT NULL,              -- 'open' | 'close'
    contracts INTEGER NOT NULL,
    credit_or_debit REAL NOT NULL,     -- positive credit for opens, positive debit for closes
    max_loss REAL,                     -- total dollars at risk (opens only)
    net_delta REAL,                    -- total $-delta contribution (opens only)
    net_vega REAL,                     -- total $-vega contribution (opens only)
    expiration TEXT,
    days_to_earnings INTEGER,
    legs_json TEXT NOT NULL,
    rationale TEXT,
    order_id TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT,
    symbol TEXT,
    decision TEXT NOT NULL,            -- 'gate_rejected' | 'llm_skipped' | 'llm_approved' | 'no_candidates'
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS iv_history (
    symbol TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    atm_iv REAL NOT NULL,
    PRIMARY KEY (symbol, observed_at)
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    snapshot_date TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    equity_peak REAL NOT NULL,
    daily_pnl_pct REAL NOT NULL,
    open_position_count INTEGER NOT NULL,
    portfolio_heat REAL NOT NULL
);
"""


@contextmanager
def connect(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_trade(
    db_path: str,
    *,
    candidate_id: str | None,
    position_group: str | None = None,
    symbol: str,
    sleeve: str,
    strategy_type: str,
    action: str,
    contracts: int,
    credit_or_debit: float,
    legs: list[dict],
    rationale: str | None,
    order_id: str | None,
    max_loss: float | None = None,
    net_delta: float | None = None,
    net_vega: float | None = None,
    expiration: str | None = None,
    days_to_earnings: int | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO trades
               (candidate_id, position_group, symbol, sleeve, strategy_type, action, contracts,
                credit_or_debit, max_loss, net_delta, net_vega, expiration, days_to_earnings,
                legs_json, rationale, order_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                candidate_id, position_group or candidate_id, symbol, sleeve, strategy_type, action, contracts,
                credit_or_debit, max_loss, net_delta, net_vega, expiration, days_to_earnings,
                json.dumps(legs), rationale, order_id,
                datetime.utcnow().isoformat(),
            ),
        )


def open_positions(db_path: str) -> list[dict]:
    """Reconstructs currently-open positions from the trades ledger: every
    'open' row whose position_group has no matching 'close' row yet. This is
    the authoritative source for portfolio state (risk gates, circuit
    breaker), rather than re-deriving it from Alpaca's raw position payload,
    since we already know the exact max_loss/delta/vega we computed and
    risk-gated at entry time.
    """
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT t.position_group, t.symbol, t.sleeve, t.strategy_type, t.contracts,
                      t.credit_or_debit, t.max_loss, t.net_delta, t.net_vega, t.expiration,
                      t.days_to_earnings, t.legs_json, t.created_at
               FROM trades t
               WHERE t.action = 'open'
                 AND NOT EXISTS (
                     SELECT 1 FROM trades c
                     WHERE c.action = 'close' AND c.position_group = t.position_group
                 )"""
        ).fetchall()
        columns = [
            "position_group", "symbol", "sleeve", "strategy_type", "contracts",
            "credit_received", "max_loss", "net_delta", "net_vega", "expiration",
            "days_to_earnings", "legs_json", "opened_at",
        ]
        return [dict(zip(columns, row)) for row in rows]


def open_position_by_group(db_path: str, position_group: str) -> dict | None:
    """Single open (not-yet-closed) position by its position_group, for the
    dashboard's manual-close action. Same reconstruction rule as
    open_positions(): an 'open' row with no matching 'close' row yet."""
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT t.position_group, t.symbol, t.sleeve, t.strategy_type, t.contracts,
                      t.credit_or_debit AS credit_received, t.max_loss, t.net_delta, t.net_vega,
                      t.expiration, t.days_to_earnings, t.legs_json, t.created_at AS opened_at
               FROM trades t
               WHERE t.action = 'open' AND t.position_group = ?
                 AND NOT EXISTS (
                     SELECT 1 FROM trades c
                     WHERE c.action = 'close' AND c.position_group = t.position_group
                 )""",
            (position_group,),
        ).fetchone()
        if row is None:
            return None
        columns = [
            "position_group", "symbol", "sleeve", "strategy_type", "contracts",
            "credit_received", "max_loss", "net_delta", "net_vega", "expiration",
            "days_to_earnings", "legs_json", "opened_at",
        ]
        return dict(zip(columns, row))


def log_decision(db_path: str, *, candidate_id: str | None, symbol: str | None, decision: str, detail: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO decisions (candidate_id, symbol, decision, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (candidate_id, symbol, decision, detail, datetime.utcnow().isoformat()),
        )


def record_iv_observation(db_path: str, *, symbol: str, observed_at: date, atm_iv: float) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO iv_history (symbol, observed_at, atm_iv) VALUES (?, ?, ?)",
            (symbol, observed_at.isoformat(), atm_iv),
        )


def iv_history_for(db_path: str, symbol: str, lookback_days: int, before: date) -> list[float]:
    with connect(db_path) as conn:
        cursor = conn.execute(
            """SELECT atm_iv FROM iv_history
               WHERE symbol = ? AND observed_at < ?
               ORDER BY observed_at DESC LIMIT ?""",
            (symbol, before.isoformat(), lookback_days),
        )
        return [row[0] for row in cursor.fetchall()]


def record_equity_snapshot(
    db_path: str,
    *,
    snapshot_date: date,
    equity: float,
    equity_peak: float,
    daily_pnl_pct: float,
    open_position_count: int,
    portfolio_heat: float,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO equity_snapshots
               (snapshot_date, equity, equity_peak, daily_pnl_pct, open_position_count, portfolio_heat)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (snapshot_date.isoformat(), equity, equity_peak, daily_pnl_pct, open_position_count, portfolio_heat),
        )


def recent_cycle_log(db_path: str, limit: int = 30) -> list[dict]:
    """Every recorded cycle check-in (ran / skipped-market-closed / errored),
    for a "is the scheduler actually alive" view -- distinct from the
    trade/gate decision feed, which only has entries when a cycle actually
    executed the screening pipeline."""
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT created_at, decision, detail FROM decisions
               WHERE decision IN ('cycle_ran', 'cycle_skipped', 'cycle_error')
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [{"created_at": r[0], "decision": r[1], "detail": r[2]} for r in rows]


def latest_equity_peak(db_path: str, fallback: float) -> float:
    with connect(db_path) as conn:
        row = conn.execute("SELECT MAX(equity_peak) FROM equity_snapshots").fetchone()
        return row[0] if row and row[0] is not None else fallback


def equity_curve(db_path: str) -> list[tuple[str, float]]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT snapshot_date, equity FROM equity_snapshots ORDER BY snapshot_date").fetchall()
        return [(r[0], r[1]) for r in rows]
