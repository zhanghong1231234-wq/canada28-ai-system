"""Read-only API for Canada28 Matrix AI V7.7 locked predictions.
Run separately from Streamlit. No betting or order-submission routes.
"""
import os, json, hmac
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Header, HTTPException
from sqlalchemy import create_engine, text

URL = os.environ.get("DATABASE_URL", "")
KEY = os.environ.get("PREDICTION_API_KEY", "")
if not URL or not KEY:
    raise RuntimeError("DATABASE_URL and PREDICTION_API_KEY required")

URL = URL.replace("postgres://", "postgresql+psycopg://", 1)
URL = URL.replace("postgresql://", "postgresql+psycopg://", 1)
engine = create_engine(URL, pool_pre_ping=True)
app = FastAPI(title="Canada28 V7.7 Read-only Prediction API")
TASKS = ["seven", "eight", "size", "parity"]


def summarize(payload, records):
    result = []
    picks = payload.get("picks") or []
    mix = payload.get("mix") or []
    if len(picks) < 1 or len(mix) != 3:
        raise HTTPException(503, "Locked prediction payload is incomplete")
    for b in range(3):
        for k, task in enumerate(TASKS):
            try:
                pick = picks[-1][b][k]
                q = mix[b][k]
                prob = sum(q[d] for d in pick) if k < 2 else sum(
                    q[d] for d in range(10)
                    if (int(d >= 5) if k == 2 else d % 2) == pick
                )
            except (IndexError, TypeError, KeyError):
                raise HTTPException(503, "Prediction task data is incomplete")
            hits = n = 0
            for p, y in records:
                try:
                    a = p["picks"][-1][b][k]
                    v = y[b]
                except (IndexError, TypeError, KeyError):
                    continue
                n += 1
                hits += int(v in a if k < 2 else a == (int(v >= 5) if k == 2 else v % 2))
            result.append({
                "ball": "ABC"[b], "task": task, "pick": pick,
                "model_probability": float(prob), "settled": n,
                "correct": hits, "incorrect": n - hits,
                "hit_rate": hits / n if n else None
            })
    return result


@app.get("/health")
def health():
    return {"ok": True, "system": "Canada28 V7.7", "mode": "read-only"}


@app.get("/v1/predictions/latest")
def latest(authorization: str = Header(default="")):
    if not hmac.compare_digest(authorization, "Bearer " + KEY):
        raise HTTPException(401, "Unauthorized")
    with engine.connect() as c:
        draw = c.execute(text(
            "SELECT issue,stamp,a,b,c FROM v720_draws ORDER BY issue DESC LIMIT 1"
        )).first()
        if not draw:
            raise HTTPException(503, "No draw data")
        target = int(draw[0]) + 1
        row = c.execute(text(
            "SELECT created,payload FROM v760_live "
            "WHERE issue=:i AND actual IS NULL"
        ), {"i": target}).first()
        if not row:
            raise HTTPException(503, "No locked prediction for next issue")
        records = c.execute(text(
            "SELECT payload,actual FROM v760_live "
            "WHERE actual IS NOT NULL AND issue<:i ORDER BY issue"
        ), {"i": target}).all()

    try:
        dt = datetime.fromisoformat(str(draw[1]).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
        expires = dt + timedelta(seconds=210)
        seconds_left = (expires - datetime.now(timezone.utc)).total_seconds()
        if not 0 < seconds_left <= 210:
            raise ValueError()
    except (ValueError, TypeError):
        raise HTTPException(503, "Draw timestamp stale or invalid")

    return {
        "system": "Canada28 Matrix AI V7.7",
        "issue": target,
        "source_issue": int(draw[0]),
        "created": row[0],
        "estimated_draw_at": expires.isoformat(),
        "locked": True,
        "latest_draw": {
            "issue": int(draw[0]), "stamp": str(draw[1]),
            "a": int(draw[2]), "b": int(draw[3]), "c": int(draw[4])
        },
        "tasks": summarize(
            json.loads(row[1]),
            [(json.loads(p), json.loads(y)) for p, y in records]
        )
    }
