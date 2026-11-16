"""Canada28 V7.8 read-only API for the local simulation monitor.
Prediction generation and settlement remain in the Streamlit worker.
"""
import os
import json
import hmac
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Header, HTTPException, Query
from sqlalchemy import create_engine, text

URL = os.environ.get("DATABASE_URL", "").strip()
KEY = os.environ.get("PREDICTION_API_KEY", "").strip()
if not URL or not KEY:
    raise RuntimeError("DATABASE_URL and PREDICTION_API_KEY are required")

URL = URL.replace("postgres://", "postgresql+psycopg://", 1)
URL = URL.replace("postgresql://", "postgresql+psycopg://", 1)
engine = create_engine(URL, pool_pre_ping=True, pool_recycle=300)
app = FastAPI(title="Canada28 V7.8 Simulation API", version="7.8")
TASKS = ["seven", "eight", "size", "parity"]


def require_api_key(authorization: str):
    if not hmac.compare_digest(authorization.strip(), "Bearer " + KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


def summarize(payload, records):
    picks = payload.get("picks") or []
    mix = payload.get("mix") or []
    if len(picks) < 1 or len(mix) != 3:
        raise ValueError("prediction payload incomplete")
    result = []
    for b in range(3):
        for k, task in enumerate(TASKS):
            pick = picks[-1][b][k]
            q = mix[b][k]
            prob = sum(float(q[d]) for d in pick) if k < 2 else sum(
                float(q[d]) for d in range(10)
                if (int(d >= 5) if k == 2 else d % 2) == int(pick)
            )
            hits = settled = 0
            for old_payload, actual in records:
                try:
                    old_pick = json.loads(old_payload)["picks"][-1][b][k]
                    value = json.loads(actual)[b]
                    settled += 1
                    hits += int(value in old_pick if k < 2 else int(old_pick) == (int(value >= 5) if k == 2 else value % 2))
                except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
            result.append({
                "ball": "ABC"[b], "task": task, "pick": pick,
                "model_probability": float(prob), "settled": settled,
                "correct": hits, "incorrect": settled - hits,
                "hit_rate": hits / settled if settled else None
            })
    return result


@app.get("/health")
def health():
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return {"ok": True, "system": "Canada28 V7.8", "database": "connected", "mode": "simulation-read-only"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {type(exc).__name__}")


@app.get("/v1/draws")
@app.get("/v1/draws/")
def draw_history(
    from_issue: int = Query(..., ge=1),
    to_issue: int = Query(..., ge=1),
    authorization: str = Header(default="")
):
    """Stored draw results for local gap recovery; maximum span is 500 issue numbers."""
    require_api_key(authorization)
    if to_issue < from_issue:
        raise HTTPException(400, "to_issue must be greater than or equal to from_issue")
    if to_issue - from_issue > 499:
        raise HTTPException(400, "A single request may cover at most 500 issue numbers")
    try:
        with engine.connect() as c:
            rows = c.execute(text(
                "SELECT issue,a,b,c,stamp FROM v720_draws "
                "WHERE issue BETWEEN :lo AND :hi ORDER BY issue"
            ), {"lo": from_issue, "hi": to_issue}).all()
        draws = [
            {"issue": int(r[0]), "a": int(r[1]), "b": int(r[2]), "c": int(r[3]), "stamp": r[4]}
            for r in rows
        ]
        return {"from_issue": from_issue, "to_issue": to_issue, "count": len(draws), "draws": draws}
    except Exception as exc:
        raise HTTPException(503, detail=f"draw query unavailable: {type(exc).__name__}")


@app.get("/v1/predictions/latest")
def latest(authorization: str = Header(default="")):
    require_api_key(authorization)
    try:
        with engine.connect() as c:
            draw = c.execute(text(
                "SELECT issue,stamp,a,b,c FROM v720_draws ORDER BY issue DESC LIMIT 1"
            )).first()
            if not draw:
                return {"system": "Canada28 Matrix AI V7.8", "status": "no_draw_data",
                        "locked": False, "tasks": []}
            target = int(draw[0]) + 1
            row = c.execute(text(
                "SELECT created,payload FROM v760_live WHERE issue=:i AND actual IS NULL"
            ), {"i": target}).first()
            records = c.execute(text(
                "SELECT payload,actual FROM v760_live "
                "WHERE actual IS NOT NULL AND issue<:i ORDER BY issue"
            ), {"i": target}).all()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database query unavailable: {type(exc).__name__}")

    latest_draw = {
        "issue": int(draw[0]), "stamp": str(draw[1] or ""),
        "a": int(draw[2]), "b": int(draw[3]), "c": int(draw[4])
    }
    if not row:
        return {
            "system": "Canada28 Matrix AI V7.8", "status": "prediction_pending",
            "issue": target, "source_issue": int(draw[0]), "locked": False,
            "latest_draw": latest_draw, "tasks": [],
            "message": "The prediction worker has not locked this issue yet. Keep automatic sync enabled."
        }

    try:
        tasks = summarize(json.loads(row[1]), records)
    except (ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        return {
            "system": "Canada28 Matrix AI V7.8", "status": "prediction_payload_invalid",
            "issue": target, "source_issue": int(draw[0]), "created": row[0],
            "locked": False, "latest_draw": latest_draw, "tasks": [],
            "message": "Stored prediction data is incomplete; regenerate it in the prediction worker."
        }

    try:
        dt = datetime.fromisoformat(str(draw[1]).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
        estimated = (dt + timedelta(seconds=210)).isoformat()
    except (ValueError, TypeError):
        estimated = None

    payload = json.loads(row[1])
    balls = {
        ball: {
            "eight_numbers": list(payload["picks"][-1][b][1]),
            "number_probabilities": [float(v) for v in payload["mix"][b][1]]
        }
        for b, ball in enumerate(("A", "B", "C"))
    }
    return {
        "system": "Canada28 Matrix AI V7.8", "status": "locked",
        "issue": target, "source_issue": int(draw[0]), "created": row[0],
        "estimated_draw_at": estimated, "locked": True,
        "latest_draw": latest_draw, "balls": balls, "tasks": tasks
    }
