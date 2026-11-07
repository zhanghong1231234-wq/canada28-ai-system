# -*- coding: utf-8 -*-
"""Canada28 Matrix AI V6.5.1
历史统计与大小/单双概率研究工具。非投注软件，不保证预测准确。
"""
import os
import sqlite3
import io
import json
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_BASE = "https://pc28.world/api/v1"
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "canada28_history.csv"
AUTO_SYNC_SECONDS = 210
DEFAULT_IMPORT = 5000


# Optional persistent PostgreSQL, configured via Streamlit Secrets DATABASE_URL.
# Without it, SQLite is local-only and may be erased on Streamlit Cloud redeploy.
def get_database_url():
    try:
        return str(st.secrets.get("DATABASE_URL", "")).strip() or os.environ.get("DATABASE_URL", "").strip()
    except Exception:
        return os.environ.get("DATABASE_URL", "").strip()


def db_connect():
    url = get_database_url()
    if url:
        import psycopg2
        conn = psycopg2.connect(url, connect_timeout=8, sslmode="require")
        return conn, True
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(DATA_DIR / "canada28_v58.sqlite3"), timeout=15), False


def db_init():
    conn, remote = db_connect()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS canada28_artifacts (
                    artifact_key TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.close()
    finally:
        conn.close()
    return remote


def db_get(key):
    conn, remote = db_connect()
    try:
        cursor = conn.cursor()
        placeholder = "%s" if remote else "?"
        cursor.execute(f"SELECT content FROM canada28_artifacts WHERE artifact_key={placeholder}", (key,))
        result = cursor.fetchone()
        cursor.close()
        return result[0] if result else None
    finally:
        conn.close()


def db_put(key, content):
    conn, remote = db_connect()
    try:
        with conn:
            cursor = conn.cursor()
            placeholder = "%s" if remote else "?"
            cursor.execute(
                "INSERT INTO canada28_artifacts (artifact_key, content, updated_at) "
                f"VALUES ({placeholder}, {placeholder}, {placeholder}) "
                "ON CONFLICT(artifact_key) DO UPDATE SET content=excluded.content, "
                "updated_at=excluded.updated_at",
                (key, content, beijing_now().isoformat(timespec="seconds")),
            )
            cursor.close()
    finally:
        conn.close()


def restore_artifacts():
    # PostgreSQL is authoritative if configured; do not silently overwrite it with local CSV.
    for filename in ("canada28_history.csv", "canada28_forecast_log.csv",
                     "calibration_state.json", "calibration_history.csv",
                     "v60_weight_history.csv", "v61_seven_digit_log.csv",
                     "v63_digit_calibration.json", "v63_digit_calibration_history.csv",
                     "v63_daily_reports.csv", "v63_digit_batches.csv",
                      "v64_six_digit_log.csv", "v64_six_calibration.json",
                      "v64_six_calibration_history.csv", "v64_six_batches.csv",
                      "v64_six_daily_reports.csv"):
        saved = db_get(filename)
        if saved is not None:
            path = DATA_DIR / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(saved, encoding="utf-8-sig" if filename.endswith(".csv") else "utf-8")


def persist_file(path):
    if path.exists():
        db_put(path.name, path.read_text(encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8"))


def ledger_audit(log, history):
    if log.empty:
        return pd.DataFrame(), []
    work = log.copy()
    work["目标期号"] = pd.to_numeric(work["目标期号"], errors="coerce")
    work["基于最新期号"] = pd.to_numeric(work["基于最新期号"], errors="coerce")
    work = work.dropna(subset=["目标期号"]).sort_values("目标期号")
    problems = []
    duplicates = int(work["目标期号"].duplicated().sum())
    if duplicates:
        problems.append(f"存在 {duplicates} 条重复目标期号")
    mismatch = work[work["目标期号"] != work["基于最新期号"] + 1]
    if len(mismatch):
        problems.append(f"存在 {len(mismatch)} 条目标期号不是基于期号+1")
    issue_values = work["目标期号"].astype(int).to_numpy()
    gaps = int(sum(max(0, int(b-a-1)) for a,b in zip(issue_values[:-1], issue_values[1:])))
    if gaps:
        problems.append(f"预测台账中有 {gaps} 个期号未记录（离线期间可能正常）")
    actual = history.set_index("期号")
    audit_rows = []
    for _, row in work.iterrows():
        issue = int(row["目标期号"])
        if issue not in actual.index:
            continue
        rec = actual.loc[issue]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[-1]
        expected_big = str(rec["大小"])
        expected_odd = str(rec["单双"])
        recorded_big = str(row.get("实际大小", ""))
        recorded_odd = str(row.get("实际单双", ""))
        if recorded_big not in ("", "nan", expected_big) or recorded_odd not in ("", "nan", expected_odd):
            audit_rows.append({"期号": issue, "问题": "台账实际结果与历史开奖不一致"})
    if audit_rows:
        problems.append(f"有 {len(audit_rows)} 条历史结果核对异常")
    return pd.DataFrame(audit_rows), problems


BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def beijing_now():
    return datetime.now(BEIJING_TZ)


def format_beijing_time(value):
    """把带时区时间转换为北京时间；无时区的源时间按北京时间解释。"""
    if value is None or str(value).strip() in ("", "nan", "NaT"):
        return ""
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Shanghai")
        else:
            ts = ts.tz_convert("Asia/Shanghai")
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def parse_beijing_time(value):
    text_value = format_beijing_time(value)
    if not text_value:
        return None
    try:
        return datetime.strptime(text_value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=BEIJING_TZ)
    except Exception:
        return None

st.set_page_config(
    page_title="Canada28 Matrix AI V6.5.1",
    page_icon="📊",
    layout="wide",
)

st.title("Canada28 Matrix AI V6.5.1")
st.caption(
    "历史数据驱动的大小 / 单双概率研究。数据来自 pc28.world；"
    "统计概率不代表能够预测随机开奖结果，也不构成投注建议。"
)


def api_json(path, params=None, timeout=15):
    response = requests.get(
        f"{API_BASE}{path}",
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Canada28-Matrix-AI/6.1"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("ok") is False:
        message = payload.get("error", {}).get("message", "API 返回异常")
        raise ValueError(str(message))
    return payload.get("data", payload)


def unwrap_draws(data):
    """兼容 draws/latest 的已文档化响应包装。"""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("draw"), dict):
        return [data["draw"]]
    if isinstance(data.get("items"), list):
        return data["items"]
    if isinstance(data.get("draws"), list):
        return data["draws"]
    return []


def normalize_rows(rows):
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            issue = int(row.get("nbr", row.get("issue")))
            a = int(row.get("n1"))
            b = int(row.get("n2"))
            c = int(row.get("n3"))
            total = int(row.get("sum", a + b + c))
            if not (0 <= a <= 9 and 0 <= b <= 9 and 0 <= c <= 9):
                continue
            if total != a + b + c or not 0 <= total <= 27:
                continue
            cleaned.append({
                "期号": issue,
                "开奖时间": str(row.get("time", "")),
                "A": a,
                "B": b,
                "C": c,
                "和值": total,
                "大小": "大" if total >= 14 else "小",
                "单双": "单" if total % 2 else "双",
            })
        except (TypeError, ValueError):
            continue
    return cleaned


def read_local():
    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)
            required = {"期号", "A", "B", "C", "和值"}
            if required.issubset(df.columns):
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "期号", "开奖时间", "A", "B", "C", "和值", "大小", "单双"
    ])


def save_local(df):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.sort_values("期号").drop_duplicates("期号", keep="last").to_csv(
        DATA_FILE, index=False, encoding="utf-8-sig"
    )
    persist_file(DATA_FILE)


def fetch_history(count):
    """按官方公开文档中的 before 游标向旧期翻页。"""
    all_rows = []
    before = None
    seen = set()
    pages = int(np.ceil(count / 200))
    for _ in range(pages):
        params = {"limit": min(200, count - len(all_rows))}
        if before is not None:
            params["before"] = before
        data = api_json("/draws", params=params)
        items = unwrap_draws(data)
        if not items:
            break
        page_rows = normalize_rows(items)
        if not page_rows:
            break
        new_rows = [r for r in page_rows if r["期号"] not in seen]
        all_rows.extend(new_rows)
        seen.update(r["期号"] for r in new_rows)
        next_before = data.get("next_before") if isinstance(data, dict) else None
        if next_before is None:
            valid_issues = [r["期号"] for r in page_rows]
            next_before = min(valid_issues) if valid_issues else None
        if next_before is None or len(items) < 1:
            break
        before = int(next_before)
        if len(all_rows) >= count:
            break
    return pd.DataFrame(all_rows).sort_values("期号").tail(count)


def fetch_latest_payload():
    """读取最新开奖及数据源提供的官方倒计时字段。"""
    data = api_json("/draws/latest")
    if not isinstance(data, dict):
        raise ValueError("最新开奖接口返回格式异常")
    return data


def fetch_latest():
    data = fetch_latest_payload()
    rows = normalize_rows(unwrap_draws(data))
    return rows[0] if rows else None


def merge_data(old, new):
    if new is None or new.empty:
        merged = old.copy()
    elif old is None or old.empty:
        merged = new.copy()
    else:
        merged = pd.concat([old, new], ignore_index=True)
    if merged.empty:
        return merged
    merged["期号"] = pd.to_numeric(merged["期号"], errors="coerce")
    merged = merged.dropna(subset=["期号"])
    merged["期号"] = merged["期号"].astype(int)
    merged = merged.sort_values("期号").drop_duplicates("期号", keep="last")
    return merged.reset_index(drop=True)


def weighted_rate(values, half_life):
    """values 为 0/1，越新的记录权重越高。"""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.5
    age = np.arange(len(arr) - 1, -1, -1, dtype=float)
    weights = np.power(0.5, age / max(float(half_life), 1.0))
    return float(np.average(arr, weights=weights))


def calculate_probabilities(df, half_life=500, prior_strength=200):
    big = (df["和值"].astype(int).to_numpy() >= 14).astype(float)
    odd = (df["和值"].astype(int).to_numpy() % 2 == 1).astype(float)

    def one_target(values):
        if len(values) == 0:
            return {"all": 0.5, "500": 0.5, "100": 0.5,
                    "weighted": 0.5, "final": 0.5}
        prior = float(prior_strength)
        p_all = (values.sum() + prior * 0.5) / (len(values) + prior)
        recent500 = values[-min(500, len(values)):]
        recent100 = values[-min(100, len(values)):]
        p500 = (recent500.sum() + prior * 0.5) / (len(recent500) + prior)
        p100 = (recent100.sum() + prior * 0.5) / (len(recent100) + prior)
        pw = (weighted_rate(values, half_life) * len(values) + prior * 0.5) / (
            len(values) + prior
        )
        # 多窗口融合，避免只追逐最近一段的随机波动。
        final = 0.30 * p_all + 0.20 * p500 + 0.20 * p100 + 0.30 * pw
        return {
            "all": float(p_all),
            "500": float(p500),
            "100": float(p100),
            "weighted": float(pw),
            "final": float(final),
        }

    return {"big": one_target(big), "odd": one_target(odd)}



# Eight distinct probability estimators. ML models train on historical lags only.
MODEL_NAMES = {
    "all": "全历史贝叶斯平滑", "500": "最近500期频率",
    "100": "最近100期频率", "weighted": "指数时间衰减",
    "markov": "一阶马尔可夫链", "pattern": "历史相似状态",
    "forest": "随机森林", "boost": "梯度提升",
    "trend50": "最近50期滚动走势",
}


def _ml_features(bits, lags=8):
    bits = np.asarray(bits, dtype=float)
    if len(bits) <= lags:
        return None, None, None
    X = np.asarray([bits[i-lags:i][::-1] for i in range(lags, len(bits))])
    y = bits[lags:].astype(int)
    next_x = bits[-lags:][::-1].reshape(1, -1)
    return X, y, next_x


def eight_model_probs(bits, half_life=500, prior=200):
    values = np.asarray(bits, dtype=float)
    n = len(values)
    base = calculate_probabilities(pd.DataFrame({"和值": np.where(values > .5, 15, 13)}), half_life, prior)["big"]
    result = {key: float(base[key]) for key in ("all", "500", "100", "weighted")}
    if n >= 20:
        # Transition probability P(next=1 | last state), shrunk toward 0.5.
        last = int(values[-1]); prev = values[:-1].astype(int)
        mask = prev == last
        hits = values[1:][mask]
        result["markov"] = float((hits.sum() + 25) / (len(hits) + 50))
        # Match the last 3 states against historical patterns, excluding current target.
        seq = values.astype(int)
        matched = [seq[i] for i in range(3, n) if np.array_equal(seq[i-3:i], seq[-3:])]
        result["pattern"] = float((sum(matched) + 25) / (len(matched) + 50))
    else:
        result["markov"] = result["pattern"] = .5
    X, y, next_x = _ml_features(values)
    if X is not None and len(X) >= 100 and len(np.unique(y)) == 2:
        # Use at most 1500 most recent training rows to keep cloud deployment responsive.
        X, y = X[-750:], y[-750:]
        try:
            from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
            forest = RandomForestClassifier(n_estimators=20, max_depth=4,
                                            min_samples_leaf=30, random_state=2026, n_jobs=1)
            boost = HistGradientBoostingClassifier(max_iter=25, max_leaf_nodes=7,
                                                    l2_regularization=10, learning_rate=.04,
                                                    random_state=2026)
            for key, model in (("forest", forest), ("boost", boost)):
                model.fit(X, y)
                raw = float(model.predict_proba(next_x)[0, list(model.classes_).index(1)])
                # Shrink ML outputs toward 0.5 to avoid misleading extreme confidence.
                result[key] = .5 + .5 * (raw - .5)
        except ImportError:
            result["forest"] = result["boost"] = .5
    else:
        result["forest"] = result["boost"] = .5
    return {k: float(np.clip(v, .01, .99)) for k, v in result.items()}


def trend_50_probs(bits):
    """Causal 50-draw descriptive trend estimate, heavily shrunk to 50%."""
    seq = np.asarray(bits, dtype=int)[-50:]
    n = len(seq)
    if n < 10:
        return .5
    # Three descriptive signals; no assumption that streaks predict reversal.
    freq = (seq.sum() + 25) / (n + 50)
    last = int(seq[-1])
    previous = seq[:-1]
    transitions = seq[1:][previous == last]
    transition = (transitions.sum() + 15) / (len(transitions) + 30)
    short = seq[-10:]
    recent = (short.sum() + 15) / (len(short) + 30)
    return float(np.clip(.5 + .5 * ((.45 * freq + .35 * transition + .20 * recent) - .5), .01, .99))


def trend_50_stats(bits):
    seq = np.asarray(bits, dtype=int)[-50:]
    streak = 0
    if len(seq):
        last = seq[-1]
        for x in seq[::-1]:
            if x != last:
                break
            streak += 1
    changes = int(np.count_nonzero(np.diff(seq))) if len(seq) > 1 else 0
    return {"n": len(seq), "yes": int(seq.sum()), "no": int(len(seq)-seq.sum()),
            "streak": streak, "last": int(seq[-1]) if len(seq) else None,
            "changes": changes}


def ensemble_probs(df, half_life=500, prior=200):
    total = df["和值"].astype(int).to_numpy()
    result = {}
    for target, bits in (("big", (total >= 14).astype(int)),
                         ("odd", (total % 2 == 1).astype(int))):
        # The two ML models are regularized; no model is declared a proven winner.
        models = eight_model_probs(bits, half_life, prior)
        models["trend50"] = trend_50_probs(bits)
        models["final"] = float(np.mean(list(models.values())))
        result[target] = models
    return result


def recommendation(p, label_yes, label_no, min_edge=0.03):
    edge = abs(p - 0.5)
    if edge < min_edge:
        return "暂不推荐", edge
    return (label_yes if p > 0.5 else label_no), edge


@st.cache_data(ttl=3600, show_spinner=False, max_entries=4)
def nine_model_backtest(df, half_life=500, prior_strength=200, tests=100):
    """Causal walk-forward evaluation including adaptive ensemble and baseline.

    ML estimators are refitted at the beginning of each 150-target block using only
    earlier rows. Their fitted parameters remain frozen within that block.
    Other estimators, including trend50, are recalculated every target draw.
    """
    if len(df) < 120:
        return None
    totals = df["和值"].astype(int).to_numpy()
    bits_map = {"big": (totals >= 14).astype(int), "odd": (totals % 2 == 1).astype(int)}
    start = max(100, len(df) - int(tests))
    warmup_start = max(100, start - 300)
    keys = list(MODEL_NAMES) + ["final", "dynamic", "baseline"]
    stats = {key: {t: {"hits": 0, "brier": 0.0} for t in bits_map} for key in keys}
    fitted = {}
    fit_errors = []
    draw_results = []
    # Only earlier resolved targets contribute to adaptive model scores.
    recent_scores = {target: {key: [] for key in MODEL_NAMES} for target in bits_map}
    for i in range(warmup_start, len(df)):
        # Train models only at the beginning of each 150-target block. This is
        # intentionally not the same as per-draw refitting in the live UI.
        if (i - warmup_start) % 150 == 0:
            fitted = {}
            for target, bits in bits_map.items():
                X, y, _ = _ml_features(bits[:i])
                if X is None or len(X) < 100 or len(np.unique(y)) < 2:
                    fitted[target] = {}
                    continue
                X, y = X[-750:], y[-750:]
                try:
                    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
                    forest = RandomForestClassifier(n_estimators=20, max_depth=4,
                        min_samples_leaf=30, random_state=2026, n_jobs=1)
                    boost = HistGradientBoostingClassifier(max_iter=25, max_leaf_nodes=7,
                        l2_regularization=10, learning_rate=.04, random_state=2026)
                    forest.fit(X, y)
                    boost.fit(X, y)
                    fitted[target] = {"forest": forest, "boost": boost}
                except ImportError:
                    fitted[target] = {}
                    fit_errors.append("scikit-learn 未安装，ML模型使用0.5基准")
        for target, bits in bits_map.items():
            prefix = bits[:i]
            # Non-ML model outputs are computed from the prefix only.
            baseline = calculate_probabilities(
                pd.DataFrame({"和值": np.where(prefix > 0, 15, 12)}),
                half_life, prior_strength)["big"]
            probs = {k: float(baseline[k]) for k in ("all", "500", "100", "weighted")}
            n = len(prefix)
            last = int(prefix[-1])
            prev = prefix[:-1]
            transitions = prefix[1:][prev == last]
            probs["markov"] = float((transitions.sum() + 25) / (len(transitions) + 50))
            matches = [prefix[j] for j in range(3, n)
                       if np.array_equal(prefix[j-3:j], prefix[-3:])]
            probs["pattern"] = float((sum(matches) + 25) / (len(matches) + 50))
            next_x = prefix[-8:][::-1].reshape(1, -1).astype(float)
            for key in ("forest", "boost"):
                model = fitted.get(target, {}).get(key)
                if model is None:
                    probs[key] = 0.5
                else:
                    raw = float(model.predict_proba(next_x)[0, list(model.classes_).index(1)])
                    probs[key] = float(np.clip(0.5 + 0.5 * (raw - 0.5), 0.01, 0.99))
            probs["trend50"] = trend_50_probs(prefix)
            probs["final"] = float(np.mean([probs[k] for k in MODEL_NAMES]))
            # Match the live V5.4 scoring/shrinkage rule, but derive all scores
            # from predictions for strictly earlier, already resolved draws.
            raw_weights = {}
            for key in MODEL_NAMES:
                history = recent_scores[target][key][-300:]
                if history:
                    hit = float(np.mean([h for h, _ in history]))
                    brier = float(np.mean([b for _, b in history]))
                    evidence = np.clip((hit - .5) * 4.0 + (.25 - brier) * 20.0, -.20, .20)
                    raw_weights[key] = float(np.exp(evidence))
                else:
                    raw_weights[key] = 1.0
            norm = sum(raw_weights.values())
            probs["dynamic"] = float(sum(
                probs[key] * (.5 * raw_weights[key] / norm + .5 / len(MODEL_NAMES))
                for key in MODEL_NAMES
            ))
            probs["baseline"] = .5
            actual = int(bits[i])
            if i >= start:
                for key in keys:
                    pred = probs[key]
                    hit = int((pred >= 0.5) == bool(actual))
                    brier = (pred - actual) ** 2
                    stats[key][target]["hits"] += hit
                    stats[key][target]["brier"] += brier
                    if key in ("pattern", "final", "dynamic", "baseline"):
                        draw_results.append({"index": i - start, "target": target,
                                             "model": key, "hit": hit, "brier": brier})
            for key in MODEL_NAMES:
                pred = probs[key]
                recent_scores[target][key].append(
                    (int((pred >= .5) == bool(actual)), (pred - actual) ** 2)
                )
    n_tests = len(df) - start
    rows = []
    for key in keys:
        rows.append({
            "模型": MODEL_NAMES.get(key, {"final": "九模型等权综合", "dynamic": "动态权重综合（严格逐期）", "baseline": "50%固定基准"}.get(key, key)),
            "大小命中率": stats[key]["big"]["hits"] / n_tests,
            "单双命中率": stats[key]["odd"]["hits"] / n_tests,
            "大小Brier分数": stats[key]["big"]["brier"] / n_tests,
            "单双Brier分数": stats[key]["odd"]["brier"] / n_tests,
        })
    return {"n": n_tests, "models": pd.DataFrame(rows),
            "draw_results": pd.DataFrame(draw_results),
            "notes": list(dict.fromkeys(fit_errors))}


# Legacy name kept for any existing callers.
rolling_backtest = nine_model_backtest


def select_models_separately(df, half_life=500, prior_strength=200, validation_tests=300):
    """Select big and odd models independently from a trailing walk-forward window.

    Hit rate is primary and Brier score breaks ties. This adaptive heuristic can
    overfit the selection window; it is not evidence of future superiority.
    """
    result = nine_model_backtest(df, half_life, prior_strength, tests=validation_tests)
    if not result or result["models"].empty:
        return {"big": "all", "odd": "all", "validation_n": 0, "table": pd.DataFrame()}
    table = result["models"]
    table = table[table["模型"].isin(list(MODEL_NAMES.values()))].copy()
    name_to_key = {v: k for k, v in MODEL_NAMES.items()}
    def choose(hit_col, brier_col):
        ranked = table.sort_values([hit_col, brier_col], ascending=[False, True], kind="stable")
        return name_to_key.get(str(ranked.iloc[0]["模型"]), "all")
    return {"big": choose("大小命中率", "大小Brier分数"),
            "odd": choose("单双命中率", "单双Brier分数"),
            "validation_n": int(result["n"]), "table": result["models"].copy(),
            "notes": result.get("notes", [])}


FORECAST_FILE = DATA_DIR / "canada28_forecast_log.csv"
CALIBRATION_STATE_FILE = DATA_DIR / "calibration_state.json"
CALIBRATION_LOG_FILE = DATA_DIR / "calibration_history.csv"
CALIBRATION_BATCH_SIZE = 50


def dynamic_model_weights(df, half_life=500, prior_strength=200, validation_tests=300):
    """Build separate adaptive weights from causal walk-forward validation.

    Scores are deliberately shrunk toward equal weights because short backtests
    are noisy. Hit rate and Brier score both contribute; this is model selection,
    not a guarantee of future edge.
    """
    result = nine_model_backtest(
        df, half_life=half_life, prior_strength=prior_strength,
        tests=validation_tests
    )
    keys = list(MODEL_NAMES)
    if not result or result["models"].empty:
        equal = {k: 1.0 / len(keys) for k in keys}
        return {"big": equal.copy(), "odd": equal.copy(),
                "validation_n": 0, "table": pd.DataFrame()}

    table = result["models"].set_index("模型")
    def weights(hit_col, brier_col):
        raw = {}
        for key, name in MODEL_NAMES.items():
            if name not in table.index:
                raw[key] = 1.0
                continue
            hit = float(table.loc[name, hit_col])
            brier = float(table.loc[name, brier_col])
            # Relative to naive 50% hit rate / 0.25 Brier; cap noisy extremes.
            evidence = np.clip((hit - 0.5) * 4.0 + (0.25 - brier) * 20.0, -0.20, 0.20)
            raw[key] = float(np.exp(evidence))
        total = sum(raw.values())
        learned = {k: v / total for k, v in raw.items()}
        # 50% learned weights + 50% equal weights: conservative regularization.
        equal = 1.0 / len(keys)
        return {k: 0.5 * learned[k] + 0.5 * equal for k in keys}

    return {
        "big": weights("大小命中率", "大小Brier分数"),
        "odd": weights("单双命中率", "单双Brier分数"),
        "validation_n": int(result["n"]),
        "table": result["models"].copy(),
        "notes": result.get("notes", []),
    }


def apply_dynamic_weights(probabilities, weight_state):
    """Apply independent big/odd weights to the nine live model probabilities."""
    output = {}
    for target in ("big", "odd"):
        weights = weight_state[target]
        output[target] = float(sum(
            float(probabilities[target][key]) * float(weights[key])
            for key in MODEL_NAMES
        ))
    return output


V60_WEIGHT_HISTORY_FILE = DATA_DIR / "v60_weight_history.csv"


def save_weight_snapshot(issue, weight_state, half_life, prior_strength):
    """Append-only, one weight snapshot per issue/target; never backfill old issues."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if V60_WEIGHT_HISTORY_FILE.exists():
        try:
            previous = pd.read_csv(V60_WEIGHT_HISTORY_FILE)
        except Exception:
            previous = pd.DataFrame()
    else:
        previous = pd.DataFrame()
    existing = set()
    if not previous.empty and {"期号", "项目"}.issubset(previous.columns):
        existing = set(zip(pd.to_numeric(previous["期号"], errors="coerce"),
                           previous["项目"].astype(str)))
    rows = []
    for target, label in (("big", "大小"), ("odd", "单双")):
        if (issue, label) in existing:
            continue
        row = {"记录时间": beijing_now().isoformat(timespec="seconds"),
               "期号": int(issue), "项目": label,
               "验证样本": int(weight_state.get("validation_n", 0)),
               "半衰期": int(half_life), "平滑强度": int(prior_strength)}
        row.update({f"权重_{key}": float(weight_state[target][key])
                    for key in MODEL_NAMES})
        rows.append(row)
    if rows:
        updated = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
        updated.to_csv(V60_WEIGHT_HISTORY_FILE, index=False, encoding="utf-8-sig")
        persist_file(V60_WEIGHT_HISTORY_FILE)


def prospective_model_window_scores(ledger, target, window):
    """Only recorded before-result V5.9+ probabilities; no reconstructed forecasts."""
    actual_col = "实际大小" if target == "big" else "实际单双"
    positive = "大" if target == "big" else "单"
    valid_actual = ("大", "小") if target == "big" else ("单", "双")
    settled = ledger[ledger[actual_col].astype(str).isin(valid_actual)].copy()
    settled["_issue"] = pd.to_numeric(settled["目标期号"], errors="coerce")
    settled = settled.dropna(subset=["_issue"]).sort_values("_issue")
    # A common sample across all models ensures fair within-window comparison.
    for key in MODEL_NAMES:
        col = f"V59_{target}_{key}_p"
        if col not in settled:
            return pd.DataFrame()
        settled[col] = pd.to_numeric(settled[col], errors="coerce")
        settled = settled[settled[col].between(0, 1)]
    settled = settled.tail(int(window))
    if settled.empty:
        return pd.DataFrame()
    y = (settled[actual_col].astype(str) == positive).astype(int)
    rows = []
    for key, name in MODEL_NAMES.items():
        p = settled[f"V59_{target}_{key}_p"]
        rows.append({"模型": name, "已核对期数": len(settled),
                     "命中率": float(((p >= .5).astype(int) == y).mean()),
                     "Brier": float(((p - y) ** 2).mean()),
                     "相对50%基准": float(((p - y) ** 2).mean() - .25)})
    rows.append({"模型": "固定50%概率基准", "已核对期数": len(settled),
                 "命中率": np.nan, "Brier": .25, "相对50%基准": 0.0})
    return pd.DataFrame(rows).sort_values("Brier")


# V6.1: independent seven-digit forecasts for the three positions.
# Keep this ledger separate from V6.0 so all existing prospective records survive.
V61_DIGIT_FILE = DATA_DIR / "v61_seven_digit_log.csv"
V61_DIGIT_COLS = ["预测时间", "目标期号", "基于最新期号"]
for _position in ("A", "B", "C"):
    V61_DIGIT_COLS += ([f"{_position}七码", f"{_position}实际号码", f"{_position}是否命中"]
                       + [f"{_position}概率{digit}" for digit in range(10)])


def _digit_model_distributions(values, half_life=500):
    """Five causal, regularized 10-class probability estimators."""
    values = np.asarray(values, dtype=int)
    n = len(values)
    uniform = np.full(10, .1, dtype=float)
    if n == 0:
        return {name: uniform.copy() for name in ("全历史", "近500期", "近100期", "时间衰减", "状态转移")}

    def smooth(sample, prior):
        counts = np.bincount(sample, minlength=10).astype(float)
        return (counts + prior / 10.0) / (len(sample) + prior)

    overall = smooth(values, 300)
    recent500 = smooth(values[-500:], 250)
    recent100 = smooth(values[-100:], 150)
    ages = np.arange(n - 1, -1, -1, dtype=float)
    weights = np.power(.5, ages / max(1., float(half_life)))
    counts = np.bincount(values, weights=weights, minlength=10)
    decay = (counts + 30 / 10.0) / (weights.sum() + 30)
    transitions = values[1:][values[:-1] == values[-1]] if n > 1 else np.array([], dtype=int)
    transition = smooth(transitions, 150)
    return {"全历史": overall, "近500期": recent500, "近100期": recent100,
            "时间衰减": decay, "状态转移": transition}


def seven_digit_probabilities(history, position, half_life=500):
    """Fit weights using strictly earlier outcomes; never use target result."""
    values = pd.to_numeric(history[position], errors="coerce").dropna().astype(int).to_numpy()
    values = values[(values >= 0) & (values <= 9)]
    names = ("全历史", "近500期", "近100期", "时间衰减", "状态转移")
    scores = {name: [] for name in names}
    # Last 100 outcomes are scored only with predictions computed BEFORE each outcome.
    # Every fifth target keeps page refreshes lightweight.
    first = max(100, len(values) - 100)
    for i in range(first, len(values), 5):
        earlier = _digit_model_distributions(values[:i], half_life)
        actual = int(values[i])
        for name in names:
            # Multiclass Brier: sum((p_k - onehot_k)^2).
            prob = earlier[name]
            scores[name].append(float(np.sum(prob ** 2) - 2 * prob[actual] + 1))
    means = {name: float(np.mean(scores[name])) if scores[name] else .9 for name in names}
    # Strong regularization: limited evidence cannot create extreme model weights.
    raw = np.array([np.exp(np.clip((.9 - means[name]) * 3, -.15, .15)) for name in names])
    learned = raw / raw.sum()
    weights = .75 / len(names) + .25 * learned
    models = _digit_model_distributions(values, half_life)
    estimate = sum(weights[j] * models[name] for j, name in enumerate(names))
    # 20% uniform mixing limits false precision on noisy draws.
    estimate = .8 * estimate + .2 * .1
    estimate = np.maximum(estimate, 0.)
    estimate /= estimate.sum()
    return estimate, {name: float(weights[j]) for j, name in enumerate(names)}, len(range(first, len(values), 5))


V63_DIGIT_STATE = DATA_DIR / "v63_digit_calibration.json"
V63_DIGIT_HISTORY = DATA_DIR / "v63_digit_calibration_history.csv"


def _digit_validation_score(values, half_life, points=30):
    """Prequential 10-class Brier; all features strictly precede target."""
    values = np.asarray(values, dtype=int)
    if len(values) < 150:
        return None
    first = max(100, len(values) - points * 5)
    losses = []
    for i in range(first, len(values), 5):
        models = _digit_model_distributions(values[:i], half_life)
        # Equal model blend is stable for small calibration samples.
        estimate = .8 * np.mean(list(models.values()), axis=0) + .2 * .1
        actual = values[i]
        losses.append(float(np.sum(estimate ** 2) - 2 * estimate[actual] + 1))
    return float(np.mean(losses)) if losses else None


def calibrate_digit_models(history, ledger, default_half_life):
    """Independently recalibrate every 50 settled prospective forecasts per ball.

    Never change existing forecasts, outcomes or their original probabilities.
    This state is independent of the big/odd calibration.
    """
    try:
        state = json.loads(V63_DIGIT_STATE.read_text(encoding="utf-8")) if V63_DIGIT_STATE.exists() else {}
    except (ValueError, OSError):
        state = {}
    state.setdefault("positions", {})
    events = []
    result = {}
    for pos in ("A", "B", "C"):
        settled = ledger[ledger[f"{pos}是否命中"].isin(["正确", "错误"])] if not ledger.empty else pd.DataFrame()
        n = len(settled)
        cfg = state["positions"].setdefault(pos, {"calibrated_count": 0, "half_life": int(default_half_life), "rounds": 0})
        old_count = int(cfg.get("calibrated_count", 0))
        chosen = int(cfg.get("half_life", default_half_life))
        # First calibration at 50 settled; subsequent at 100, 150, ...
        if n >= old_count + 50:
            previous_half = chosen
            values = pd.to_numeric(history[pos], errors="coerce").dropna().astype(int).to_numpy()
            values = values[(values >= 0) & (values <= 9)]
            scores = {half: _digit_validation_score(values, half) for half in (200, 500, 1000)}
            valid = {half: score for half, score in scores.items() if score is not None}
            before = _digit_validation_score(values, chosen)
            # Small differences on noisy draws are not evidence: require a margin.
            if valid:
                best = min(valid, key=valid.get)
                if before is None or valid[best] + .002 < before:
                    chosen = best
                cfg["half_life"] = chosen
                cfg["calibrated_count"] = (n // 50) * 50
                cfg["rounds"] = max(int(cfg.get("rounds", 0)), int(cfg["calibrated_count"]) // 50)
                events.append({"校准时间(北京时间)": beijing_now().isoformat(timespec="seconds"),
                               "球位": pos, "已核对期数": n, "本轮累计核对": cfg["calibrated_count"],
                               "校准前半衰期": previous_half,
                               "选定半衰期": chosen,
                               "原参数验证Brier": round(before, 6) if before is not None else None,
                               "选定参数验证Brier": round(valid.get(chosen, before), 6) if valid.get(chosen, before) is not None else None,
                               "理论均匀模型Brier": .9,
                               "备注": "仅影响后续新预测；已保存记录不变"})
        result[pos] = {"half_life": chosen, "settled": n,
                       "rounds": int(cfg.get("rounds", 0)),
                       "remaining": max(0, 50 - (n - int(cfg.get("calibrated_count", 0))))}
    if events:
        extra = pd.DataFrame(events)
        if V63_DIGIT_HISTORY.exists():
            try:
                extra = pd.concat([pd.read_csv(V63_DIGIT_HISTORY), extra], ignore_index=True)
            except (OSError, pd.errors.EmptyDataError):
                pass
        extra.to_csv(V63_DIGIT_HISTORY, index=False, encoding="utf-8-sig")
        persist_file(V63_DIGIT_HISTORY)
        V63_DIGIT_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        persist_file(V63_DIGIT_STATE)
    return result


V63_DAILY_REPORT = DATA_DIR / "v63_daily_reports.csv"
V63_DIGIT_BATCHES = DATA_DIR / "v63_digit_batches.csv"


def build_digit_reports(ledger, now=None):
    """Persist settled 50-forecast blocks and one Beijing-date report per completed day.

    Runs only while the Streamlit app is active; never assumes missing draws mean
    a confirmed official closure. Incomplete days are reported the next day.
    """
    now = now or beijing_now()
    if ledger.empty:
        return {"batches": 0, "daily": 0, "pending": 0}
    frame = ledger.copy()
    frame["_issue"] = pd.to_numeric(frame["目标期号"], errors="coerce")
    frame = frame.dropna(subset=["_issue"]).sort_values("_issue").drop_duplicates("_issue", keep="first")
    frame["_prediction_date"] = pd.to_datetime(frame["预测时间"], errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
    # A naive timestamp from an earlier version is Beijing local time, not UTC.
    def local_date(value):
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("Asia/Shanghai")
            else:
                ts = ts.tz_convert("Asia/Shanghai")
            return ts.strftime("%Y-%m-%d")
        except Exception:
            return ""
    frame["_prediction_date"] = frame["预测时间"].map(local_date)
    batch_rows, daily_rows = [], []
    for pos in ("A", "B", "C"):
        settled = frame[frame[f"{pos}是否命中"].isin(("正确", "错误"))].copy()
        for start in range(0, len(settled) - 49, 50):
            block = settled.iloc[start:start + 50]
            hits = int((block[f"{pos}是否命中"] == "正确").sum())
            batch_rows.append({"球位": pos, "档次": start // 50 + 1,
                               "起始期号": int(block.iloc[0]["_issue"]),
                               "截止期号": int(block.iloc[-1]["_issue"]),
                               "已核对": 50, "命中": hits, "覆盖率": hits / 50,
                               "理论基准": .7})
        # Daily report is finalized after 19:05 Beijing time or on a later day;
        # a finalization never prevents incoming draws from being processed.
        for date, group in settled.groupby("_prediction_date"):
            if not date:
                continue
            today = now.strftime("%Y-%m-%d")
            if date > today or (date == today and (now.hour, now.minute) < (19, 5)):
                continue
            hits = int((group[f"{pos}是否命中"] == "正确").sum())
            daily_rows.append({"预测日期(北京时间)": date, "球位": pos,
                               "已核对": len(group), "命中": hits,
                               "覆盖率": hits / len(group), "理论基准": .7,
                               "最后核对期号": int(group["_issue"].max()),
                               "生成时间(北京时间)": now.isoformat(timespec="seconds")})
    # Rebuild derived tables from authoritative prospective ledger to avoid duplicates.
    for path, rows in ((V63_DIGIT_BATCHES, batch_rows), (V63_DAILY_REPORT, daily_rows)):
        if rows:
            output = pd.DataFrame(rows)
            # Do not rewrite persistent artifacts when no new verified result exists.
            current = None
            if path.exists():
                try:
                    current = pd.read_csv(path)
                except (OSError, pd.errors.EmptyDataError):
                    pass
            compare_cols = [c for c in output.columns if c != "生成时间(北京时间)"]
            if current is None or not set(compare_cols).issubset(current.columns) or not output[compare_cols].astype(str).equals(current[compare_cols].astype(str)):
                output.to_csv(path, index=False, encoding="utf-8-sig")
                persist_file(path)
    return {"batches": len(batch_rows), "daily": len(daily_rows),
            "pending": sum(max(0, len(frame[frame[f"{pos}是否命中"].isin(("正确", "错误"))]) % 50) for pos in ("A", "B", "C"))}


def seven_digit_forecast(history, half_life=500, per_position=None):
    result = {}
    for position in ("A", "B", "C"):
        probabilities, weights, validation_n = seven_digit_probabilities(
            history, position, (per_position or {}).get(position, {}).get("half_life", half_life))
        ranking = sorted(range(10), key=lambda digit: (-probabilities[digit], digit))
        result[position] = {"probs": probabilities, "seven": "".join(map(str, ranking[:7])),
                            "weights": weights, "validation_n": validation_n,
                            "coverage_estimate": float(sum(probabilities[d] for d in ranking[:7]))}
    return result


def update_seven_digit_ledger(history, prediction):
    """Write once per future issue, then only settle against observed draw numbers."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log = pd.read_csv(V61_DIGIT_FILE, dtype={f"{p}七码": str for p in ("A", "B", "C")})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        log = pd.DataFrame(columns=V61_DIGIT_COLS)
    for col in V61_DIGIT_COLS:
        if col not in log.columns:
            log[col] = ""
    log["目标期号"] = pd.to_numeric(log["目标期号"], errors="coerce")
    latest = int(history["期号"].max())
    target = latest + 1
    changed = False
    if not (log["目标期号"] == target).any():
        record = {"预测时间": beijing_now().isoformat(timespec="seconds"),
                  "目标期号": target, "基于最新期号": latest}
        for position in ("A", "B", "C"):
            record[f"{position}七码"] = prediction[position]["seven"]
            record[f"{position}实际号码"] = ""
            record[f"{position}是否命中"] = "等待开奖"
            for digit, p in enumerate(prediction[position]["probs"]):
                record[f"{position}概率{digit}"] = round(float(p), 8)
        log = pd.concat([log, pd.DataFrame([record])], ignore_index=True)
        changed = True
    actuals = history.set_index("期号")
    for idx, record in log.iterrows():
        issue = record["目标期号"]
        if pd.isna(issue) or int(issue) not in actuals.index:
            continue
        draw = actuals.loc[int(issue)]
        if isinstance(draw, pd.DataFrame):
            draw = draw.iloc[-1]
        for position in ("A", "B", "C"):
            actual = int(draw[position])
            seven = str(record[f"{position}七码"])
            # Only valid original seven-digit forecasts are scored.
            if len(seven) != 7 or len(set(seven)) != 7 or not seven.isdigit():
                continue
            verdict = "正确" if str(actual) in seven else "错误"
            if str(record[f"{position}实际号码"]) != str(actual) or str(record[f"{position}是否命中"]) != verdict:
                log.at[idx, f"{position}实际号码"] = actual
                log.at[idx, f"{position}是否命中"] = verdict
                changed = True
    if changed:
        log = log[V61_DIGIT_COLS].sort_values("目标期号").drop_duplicates("目标期号", keep="first")
        log.to_csv(V61_DIGIT_FILE, index=False, encoding="utf-8-sig")
        persist_file(V61_DIGIT_FILE)
    return log


# V6.4: independent prospective six-of-ten forecasts. The seven-digit ledger is untouched.
V64_SIX_FILE = DATA_DIR / "v64_six_digit_log.csv"
V64_SIX_STATE = DATA_DIR / "v64_six_calibration.json"
V64_SIX_HISTORY = DATA_DIR / "v64_six_calibration_history.csv"
V64_SIX_BATCHES = DATA_DIR / "v64_six_batches.csv"
V64_SIX_DAILY = DATA_DIR / "v64_six_daily_reports.csv"
V64_SIX_COLS = ["预测时间", "目标期号", "基于最新期号"]
for _pos in ("A", "B", "C"):
    V64_SIX_COLS += ([f"{_pos}六码", f"{_pos}实际号码", f"{_pos}是否命中",
                      f"{_pos}锁定模型", f"{_pos}锁定估计覆盖率", f"{_pos}锁定半衰期"]
                     + [f"{_pos}概率{digit}" for digit in range(10)])


def update_six_digit_ledger(history, predictions):
    """Lock a first six-digit forecast for a future issue; settle only actual draws."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log = pd.read_csv(V64_SIX_FILE, dtype={f"{p}六码": str for p in ("A", "B", "C")})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        log = pd.DataFrame(columns=V64_SIX_COLS)
    for col in V64_SIX_COLS:
        if col not in log.columns:
            log[col] = ""
    log["目标期号"] = pd.to_numeric(log["目标期号"], errors="coerce")
    latest = int(history["期号"].max())
    target = latest + 1
    changed = False
    if not (log["目标期号"] == target).any():
        record = {"预测时间": beijing_now().isoformat(timespec="seconds"),
                  "目标期号": target, "基于最新期号": latest}
        for pos in ("A", "B", "C"):
            record[f"{pos}六码"] = predictions[pos]["six"]
            record[f"{pos}实际号码"] = ""
            record[f"{pos}是否命中"] = "等待开奖"
            record[f"{pos}锁定模型"] = predictions[pos].get("selected_model", "")
            record[f"{pos}锁定估计覆盖率"] = predictions[pos].get("coverage_estimate", "")
            record[f"{pos}锁定半衰期"] = predictions[pos].get("half_life", "")
            for digit, prob in enumerate(predictions[pos]["probs"]):
                record[f"{pos}概率{digit}"] = round(float(prob), 8)
        log = pd.concat([log, pd.DataFrame([record])], ignore_index=True)
        changed = True
    actuals = history.set_index("期号")
    for idx, record in log.iterrows():
        issue = record["目标期号"]
        if pd.isna(issue) or int(issue) not in actuals.index:
            continue
        draw = actuals.loc[int(issue)]
        if isinstance(draw, pd.DataFrame):
            draw = draw.iloc[-1]
        for pos in ("A", "B", "C"):
            digits = str(record[f"{pos}六码"])
            if len(digits) != 6 or not digits.isdigit() or len(set(digits)) != 6:
                continue
            actual = int(draw[pos])
            verdict = "正确" if str(actual) in digits else "错误"
            if str(record[f"{pos}实际号码"]) != str(actual) or str(record[f"{pos}是否命中"]) != verdict:
                log.at[idx, f"{pos}实际号码"] = actual
                log.at[idx, f"{pos}是否命中"] = verdict
                changed = True
    if changed:
        log = log[V64_SIX_COLS].sort_values("目标期号").drop_duplicates("目标期号", keep="first")
        log.to_csv(V64_SIX_FILE, index=False, encoding="utf-8-sig")
        persist_file(V64_SIX_FILE)
    return log


def calibrate_six_models(history, ledger, default_half_life):
    """Separate six-digit calibration after each 50 settled prospective forecasts."""
    try:
        state = json.loads(V64_SIX_STATE.read_text(encoding="utf-8")) if V64_SIX_STATE.exists() else {}
    except (OSError, ValueError):
        state = {}
    state.setdefault("positions", {})
    result, events = {}, []
    for pos in ("A", "B", "C"):
        settled = ledger[ledger[f"{pos}是否命中"].isin(("正确", "错误"))] if not ledger.empty else pd.DataFrame()
        n = len(settled)
        cfg = state["positions"].setdefault(pos, {"calibrated_count": 0,
                            "half_life": int(default_half_life), "rounds": 0})
        calibrated = int(cfg.get("calibrated_count", 0))
        chosen = int(cfg.get("half_life", default_half_life))
        if n >= calibrated + 50:
            values = pd.to_numeric(history[pos], errors="coerce").dropna().astype(int).to_numpy()
            values = values[(values >= 0) & (values <= 9)]
            scores = {h: _digit_validation_score(values, h) for h in (200, 500, 1000)}
            valid = {h: score for h, score in scores.items() if score is not None}
            old_score = _digit_validation_score(values, chosen)
            if valid:
                best = min(valid, key=valid.get)
                if old_score is None or valid[best] + .002 < old_score:
                    chosen = best
            # Count only settled, pre-recorded six-digit forecasts; do not backfill.
            cfg["calibrated_count"] = (n // 50) * 50
            cfg["rounds"] = cfg["calibrated_count"] // 50
            cfg["half_life"] = chosen
            events.append({"校准时间(北京时间)": beijing_now().isoformat(timespec="seconds"),
                           "球位": pos, "已核对期数": n,
                           "本轮累计核对": cfg["calibrated_count"], "选定半衰期": chosen,
                           "验证Brier": valid.get(chosen, old_score), "均匀基准Brier": .9})
        result[pos] = {"half_life": chosen, "settled": n,
                       "rounds": int(cfg.get("rounds", 0)),
                       "remaining": max(0, 50 - (n - int(cfg.get("calibrated_count", 0))))}
    if events:
        extra = pd.DataFrame(events)
        if V64_SIX_HISTORY.exists():
            try:
                extra = pd.concat([pd.read_csv(V64_SIX_HISTORY), extra], ignore_index=True)
            except (OSError, pd.errors.EmptyDataError):
                pass
        extra.to_csv(V64_SIX_HISTORY, index=False, encoding="utf-8-sig")
        persist_file(V64_SIX_HISTORY)
        V64_SIX_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        persist_file(V64_SIX_STATE)
    return result


def build_six_reports(ledger, now=None):
    """Six-digit 50-settled batches and Beijing daily reports; separate from seven-digit."""
    now = now or beijing_now()
    if ledger.empty:
        return {"batches": 0, "daily": 0}
    frame = ledger.copy()
    frame["_issue"] = pd.to_numeric(frame["目标期号"], errors="coerce")
    frame = frame.dropna(subset=["_issue"]).sort_values("_issue").drop_duplicates("_issue", keep="first")
    def date_bj(value):
        try:
            ts = pd.Timestamp(value)
            ts = ts.tz_localize("Asia/Shanghai") if ts.tzinfo is None else ts.tz_convert("Asia/Shanghai")
            return ts.strftime("%Y-%m-%d")
        except Exception:
            return ""
    frame["_date"] = frame["预测时间"].map(date_bj)
    batches, daily = [], []
    for pos in ("A", "B", "C"):
        settled = frame[frame[f"{pos}是否命中"].isin(("正确", "错误"))]
        for start in range(0, len(settled) - 49, 50):
            block = settled.iloc[start:start + 50]
            hits = int((block[f"{pos}是否命中"] == "正确").sum())
            batches.append({"球位": pos, "档次": start // 50 + 1,
                            "起始期号": int(block.iloc[0]["_issue"]),
                            "截止期号": int(block.iloc[-1]["_issue"]),
                            "已核对": 50, "命中": hits, "覆盖率": hits / 50, "理论基准": .6})
        for date, group in settled.groupby("_date"):
            if not date or date > now.strftime("%Y-%m-%d"):
                continue
            if date == now.strftime("%Y-%m-%d") and (now.hour, now.minute) < (19, 5):
                continue
            hits = int((group[f"{pos}是否命中"] == "正确").sum())
            daily.append({"预测日期(北京时间)": date, "球位": pos, "已核对": len(group),
                          "命中": hits, "覆盖率": hits / len(group), "理论基准": .6,
                          "最后核对期号": int(group["_issue"].max()),
                          "生成时间(北京时间)": now.isoformat(timespec="seconds")})
    for path, rows in ((V64_SIX_BATCHES, batches), (V64_SIX_DAILY, daily)):
        if not rows:
            continue
        output = pd.DataFrame(rows)
        old = None
        if path.exists():
            try:
                old = pd.read_csv(path)
            except (OSError, pd.errors.EmptyDataError):
                pass
        cols = [c for c in output.columns if c != "生成时间(北京时间)"]
        if old is None or not set(cols).issubset(old.columns) or not output[cols].astype(str).equals(old[cols].astype(str)):
            output.to_csv(path, index=False, encoding="utf-8-sig")
            persist_file(path)
    return {"batches": len(batches), "daily": len(daily)}


# V6.5: six-only causal model selection. No seven-digit functions or data are changed.
SIX_MODEL_NAMES = ("全历史", "近500期", "近100期", "时间衰减", "状态转移", "五模型等权")


def _six_rank(probabilities):
    return sorted(range(10), key=lambda digit: (-float(probabilities[digit]), digit))[:6]


def _six_candidates(values, half_life):
    models = _digit_model_distributions(values, half_life)
    # Shrink each candidate toward the uniform 10-class prior.
    candidates = {name: .85 * probs + .15 * .1 for name, probs in models.items()}
    candidates["五模型等权"] = .85 * np.mean(list(models.values()), axis=0) + .15 * .1
    return candidates


def _six_select_model(values, half_life, points=180):
    """Strictly causal validation: choose 6 before observing each target.

    A selection threshold and uniform fallback reduce chasing random differences.
    The result is descriptive, not a guarantee of an advantage over 60%.
    """
    values = np.asarray(values, dtype=int)
    first = max(150, len(values) - points * 2)
    score = {name: [0, 0] for name in SIX_MODEL_NAMES}
    for i in range(first, len(values), 2):
        candidates = _six_candidates(values[:i], half_life)
        actual = int(values[i])
        for name in SIX_MODEL_NAMES:
            score[name][0] += int(actual in _six_rank(candidates[name]))
            score[name][1] += 1
    count = score["五模型等权"][1]
    if count < 80:
        return "五模型等权", score, "样本不足，采用等权模型"
    # Compare against a common default and require >= 3 percentage point gain
    # in held-out causal hit rate; selection itself can overfit this sample.
    baseline_rate = score["五模型等权"][0] / count
    best = max(SIX_MODEL_NAMES, key=lambda name: (score[name][0] / count, -SIX_MODEL_NAMES.index(name)))
    best_rate = score[best][0] / count
    if best != "五模型等权" and best_rate >= baseline_rate + .03 and best_rate >= .60 + .03:
        return best, score, "历史滚动验证选定；未来不保证保持优势"
    return "五模型等权", score, "差距不足，采用稳健等权模型"


def six_digit_forecast(history, per_position):
    result = {}
    for pos in ("A", "B", "C"):
        values = pd.to_numeric(history[pos], errors="coerce").dropna().astype(int).to_numpy()
        values = values[(values >= 0) & (values <= 9)]
        half_life = int(per_position[pos]["half_life"])
        selected, scores, reason = _six_select_model(values, half_life)
        candidates = _six_candidates(values, half_life)
        probs = np.asarray(candidates[selected], dtype=float)
        probs = probs / probs.sum()
        ranking = _six_rank(probs)
        n = scores[selected][1]
        result[pos] = {
            "six": "".join(map(str, ranking)), "probs": probs,
            "weights": {name: (1.0 if name == selected else 0.0) for name in SIX_MODEL_NAMES},
            "validation_n": n, "coverage_estimate": float(sum(probs[d] for d in ranking)),
            "selected_model": selected, "selection_reason": reason,
            "half_life": half_life,
            "validation_scores": {name: (hits / total if total else None)
                                  for name, (hits, total) in scores.items()},
        }
    return result


def update_forecast_ledger(df, probabilities):
    """每个目标期只保存首次预测；开奖后分别核对单双与大小。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "预测时间", "目标期号", "基于最新期号",
        "预测单双", "单双概率单", "实际单双", "单双结果",
        "预测大小", "大小概率大", "实际大小", "大小结果",
        "相似模型大小概率大", "相似模型单双概率单",
        "相似模型大小预测", "相似模型单双预测",
        "相似模型大小结果", "相似模型单双结果"
    ]
    # New model-specific probabilities are captured only before the next draw.
    model_cols = [f"V59_{t}_{key}_p" for t in ("big", "odd") for key in MODEL_NAMES]
    cols += model_cols
    try:
        log = pd.read_csv(FORECAST_FILE)
    except Exception:
        log = pd.DataFrame(columns=cols)

    # 兼容 V4.7 旧日志：先从原概率列恢复分类，再补齐新列。
    if "预测单双" not in log.columns and "单双概率单" in log.columns:
        log["预测单双"] = np.where(
            pd.to_numeric(log["单双概率单"], errors="coerce") >= 0.5, "单", "双"
        )
    if "预测大小" not in log.columns and "大小概率大" in log.columns:
        log["预测大小"] = np.where(
            pd.to_numeric(log["大小概率大"], errors="coerce") >= 0.5, "大", "小"
        )
    for col in cols:
        if col not in log.columns:
            log[col] = ""
    for idx, row in log.iterrows():
        if str(row.get("实际单双", "")) in ("单", "双"):
            log.at[idx, "单双结果"] = (
                "正确" if str(row.get("预测单双", "")) == str(row["实际单双"]) else "错误"
            )
        elif not str(row.get("单双结果", "")):
            log.at[idx, "单双结果"] = "等待开奖"
        if str(row.get("实际大小", "")) in ("大", "小"):
            log.at[idx, "大小结果"] = (
                "正确" if str(row.get("预测大小", "")) == str(row["实际大小"]) else "错误"
            )
        elif not str(row.get("大小结果", "")):
            log.at[idx, "大小结果"] = "等待开奖"

    current = int(df["期号"].max())
    target = current + 1
    existing_targets = pd.to_numeric(log["目标期号"], errors="coerce")
    if not (existing_targets == target).any():
        p_odd = float(probabilities["odd"]["final"])
        p_big = float(probabilities["big"]["final"])
        pattern_big = float(probabilities["big"]["pattern"])
        pattern_odd = float(probabilities["odd"]["pattern"])
        log = pd.concat([log, pd.DataFrame([{
            "预测时间": beijing_now().isoformat(timespec="seconds"),
            "目标期号": target,
            "基于最新期号": current,
            "预测单双": "单" if p_odd >= 0.5 else "双",
            "单双概率单": round(p_odd, 6),
            "实际单双": "",
            "单双结果": "等待开奖",
            "预测大小": "大" if p_big >= 0.5 else "小",
            "大小概率大": round(p_big, 6),
            "实际大小": "",
            "大小结果": "等待开奖",
            "相似模型大小概率大": round(pattern_big, 6),
            "相似模型单双概率单": round(pattern_odd, 6),
            "相似模型大小预测": "大" if pattern_big >= 0.5 else "小",
            "相似模型单双预测": "单" if pattern_odd >= 0.5 else "双",
            "相似模型大小结果": "等待开奖",
            "相似模型单双结果": "等待开奖",
            **{f"V59_{t}_{key}_p": round(float(probabilities[t][key]), 8)
               for t in ("big", "odd") for key in MODEL_NAMES},
        }])], ignore_index=True)

    actual = df.set_index("期号")[["大小", "单双"]]
    for idx, row in log.iterrows():
        try:
            issue = int(float(row["目标期号"]))
            if issue not in actual.index:
                continue
            actual_big = str(actual.loc[issue, "大小"])
            actual_odd = str(actual.loc[issue, "单双"])
            log.at[idx, "实际大小"] = actual_big
            log.at[idx, "实际单双"] = actual_odd
            log.at[idx, "大小结果"] = (
                "正确" if str(row["预测大小"]) == actual_big else "错误"
            )
            log.at[idx, "单双结果"] = (
                "正确" if str(row["预测单双"]) == actual_odd else "错误"
            )
            # 旧版记录不补造相似模型的事前预测。
            if str(row.get("相似模型大小预测", "")) in ("大", "小"):
                log.at[idx, "相似模型大小结果"] = (
                    "正确" if str(row["相似模型大小预测"]) == actual_big else "错误"
                )
            if str(row.get("相似模型单双预测", "")) in ("单", "双"):
                log.at[idx, "相似模型单双结果"] = (
                    "正确" if str(row["相似模型单双预测"]) == actual_odd else "错误"
                )
        except (TypeError, ValueError, KeyError):
            continue

    log = log[cols].sort_values("目标期号")
    log.to_csv(FORECAST_FILE, index=False, encoding="utf-8-sig")
    persist_file(FORECAST_FILE)
    return log


def evaluate_parameter_set(df, half_life, prior_strength, tests=50):
    """对最近一批历史开奖做严格时间顺序评估，返回大小/单双平均 Brier。"""
    if len(df) < 120:
        return None
    start = max(100, len(df) - tests)
    scores = []
    for i in range(start, len(df)):
        p = calculate_probabilities(df.iloc[:i], half_life, prior_strength)
        total = int(df.iloc[i]["和值"])
        actual_big = float(total >= 14)
        actual_odd = float(total % 2 == 1)
        scores.append(
            (float(p["big"]["final"]) - actual_big) ** 2
            + (float(p["odd"]["final"]) - actual_odd) ** 2
        )
    return float(np.mean(scores)) if scores else None


def calibrate_every_50(df, fallback_half_life, fallback_prior):
    """每累计50条新开奖记录，比较参数组合并校准；历史预测日志不回写。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current_issue = int(df["期号"].max())
    if CALIBRATION_STATE_FILE.exists():
        try:
            state = json.loads(CALIBRATION_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    else:
        state = {}

    if not state:
        state = {
            "last_calibrated_issue": current_issue,
            "half_life": int(fallback_half_life),
            "prior_strength": int(fallback_prior),
            "completed_batches": 0,
        }

    last_issue = int(state.get("last_calibrated_issue", current_issue))
    active_half = int(state.get("half_life", fallback_half_life))
    active_prior = int(state.get("prior_strength", fallback_prior))
    batches = 0
    calibration_rows = []

    new_rows = df[df["期号"].astype(int) > last_issue].sort_values("期号")
    while len(new_rows) >= CALIBRATION_BATCH_SIZE:
        batch = new_rows.iloc[:CALIBRATION_BATCH_SIZE]
        batch_end_issue = int(batch.iloc[-1]["期号"])
        history_upto_batch = df[df["期号"].astype(int) <= batch_end_issue].reset_index(drop=True)

        candidates = []
        for half in (200, 500, 1000):
            for prior in (50, 200, 400):
                score = evaluate_parameter_set(
                    history_upto_batch, half, prior, tests=CALIBRATION_BATCH_SIZE
                )
                if score is not None:
                    candidates.append((score, half, prior))

        if candidates:
            score, active_half, active_prior = min(candidates, key=lambda x: x[0])
            calibration_rows.append({
                "校准完成时间": beijing_now().isoformat(timespec="seconds"),
                "本轮截止期号": batch_end_issue,
                "本轮新开奖数": CALIBRATION_BATCH_SIZE,
                "选定半衰期": active_half,
                "选定平滑强度": active_prior,
                "时间顺序测试综合Brier": round(float(score), 6),
            })
            batches += 1
            last_issue = batch_end_issue
            new_rows = new_rows.iloc[CALIBRATION_BATCH_SIZE:]
        else:
            break

    if calibration_rows:
        hist = pd.DataFrame(calibration_rows)
        if CALIBRATION_LOG_FILE.exists():
            try:
                old_hist = pd.read_csv(CALIBRATION_LOG_FILE)
                hist = pd.concat([old_hist, hist], ignore_index=True)
            except Exception:
                pass
        hist.to_csv(CALIBRATION_LOG_FILE, index=False, encoding="utf-8-sig")
        persist_file(CALIBRATION_LOG_FILE)

    state.update({
        "last_calibrated_issue": last_issue,
        "half_life": active_half,
        "prior_strength": active_prior,
        "completed_batches": int(state.get("completed_batches", 0)) + batches,
    })
    CALIBRATION_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    persist_file(CALIBRATION_STATE_FILE)
    remaining = max(0, CALIBRATION_BATCH_SIZE - len(new_rows))
    status = (
        f"本次完成 {batches} 轮50期开奖校准"
        if batches else f"距离下次校准还差 {remaining} 期新开奖"
    )
    return active_half, active_prior, state, status


# Sidebar controls
with st.sidebar:
    st.header("系统设置")
    import_count = st.selectbox("首次同步历史期数", [1000, 3000, 5000], index=2)
    half_life = st.slider("初始近期权重半衰期（期）", 100, 2000, 500, 50)
    prior_strength = st.slider("初始先验平滑强度", 0, 1000, 200, 25)
    min_edge = st.slider("概率差参考门槛（仅显示，不过滤）", 1, 10, 3) / 100
    auto_sync = st.checkbox("开启自动同步（每210秒）", value=True)
    st.caption("网页运行时每5秒检查最新开奖，每210秒补充历史。每50期校准统计平滑参数；大小与单双分别采用九模型动态权重。")
    st.caption("数据来源：pc28.world（第三方只读数据接口）")

try:
    persistent_remote = db_init()
    restore_artifacts()
except Exception as exc:
    st.error(f"数据库连接失败：{type(exc).__name__}。请检查 DATABASE_URL、网络及数据库权限。")
    st.stop()

with st.sidebar:
    if persistent_remote:
        st.success("持久化存储：PostgreSQL 已连接")
    else:
        st.warning("当前使用本地 SQLite；Streamlit Cloud 重启可能丢失数据")

if "history_df" not in st.session_state:
    st.session_state.history_df = read_local()
if "last_sync" not in st.session_state:
    st.session_state.last_sync = 0.0
if "sync_message" not in st.session_state:
    st.session_state.sync_message = "尚未执行本次会话同步"


def perform_initial_sync():
    with st.spinner(f"正在同步最近 {import_count} 期开奖数据，请稍候…"):
        incoming = fetch_history(import_count)
        merged = merge_data(st.session_state.history_df, incoming)
        st.session_state.history_df = merged
        save_local(merged)
        st.session_state.last_sync = time.time()
        st.session_state.sync_message = f"历史同步完成：当前保存 {len(merged)} 期"


if st.session_state.history_df.empty:
    try:
        perform_initial_sync()
    except Exception as exc:
        st.error(f"首次同步失败：{exc}")
        st.info("请稍后刷新页面，或检查 pc28.world 接口是否可访问。")
# 轻量检查最新期号每5秒执行一次；发现新期开奖立即写入历史并触发页面重绘。
# 完整历史补充/维护仍按210秒执行，避免把210秒误当成最新开奖的等待时间。
@st.fragment(run_every=5 if auto_sync else None)
def sync_clock():
    if auto_sync and not st.session_state.history_df.empty:
        try:
            latest = fetch_latest()
            current_max = int(st.session_state.history_df["期号"].max())

            if latest and int(latest["期号"]) > current_max:
                # 先将接口最新一期合并进本地历史，随后主页面会自动核对预测台账。
                recent = normalize_rows(unwrap_draws(api_json("/draws", {"limit": 200})))
                latest_df = pd.DataFrame(recent) if recent else pd.DataFrame([latest])
                st.session_state.history_df = merge_data(
                    st.session_state.history_df, latest_df
                )
                save_local(st.session_state.history_df)
                st.session_state.sync_message = (
                    f"发现新期开奖，已同步至第 {int(latest['期号'])} 期"
                )
                st.session_state.last_sync = time.time()
                st.rerun(scope="app")

            elapsed = time.time() - st.session_state.last_sync
            if elapsed >= AUTO_SYNC_SECONDS:
                # 每210秒补充最近200期，确保历史数据与最新期号连续。
                recent = normalize_rows(
                    unwrap_draws(api_json("/draws", {"limit": 200}))
                )
                if recent:
                    st.session_state.history_df = merge_data(
                        st.session_state.history_df, pd.DataFrame(recent)
                    )
                    save_local(st.session_state.history_df)
                st.session_state.last_sync = time.time()
                st.session_state.sync_message = (
                    f"已完成210秒历史同步；当前最新期号第 "
                    f"{int(st.session_state.history_df['期号'].max())} 期"
                )
                if int(st.session_state.history_df["期号"].max()) > current_max:
                    st.rerun(scope="app")
        except Exception as exc:
            st.session_state.sync_message = f"自动同步暂时失败：{exc}"
    st.caption(
        f"最新期号检查：每5秒｜完整历史同步：每210秒｜{st.session_state.sync_message}"
    )

sync_clock()

df = st.session_state.history_df.copy()
if not df.empty:
    df["期号"] = pd.to_numeric(df["期号"], errors="coerce")
    df["和值"] = pd.to_numeric(df["和值"], errors="coerce")
    df = df.dropna(subset=["期号", "和值"]).sort_values("期号").reset_index(drop=True)

if df.empty:
    st.warning("目前没有可用历史数据。请检查网络或稍后重试。")
    st.stop()

latest_issue = int(df["期号"].max())
latest_time = str(df.iloc[-1].get("开奖时间", ""))

# 自动校准参数：首次运行以侧边栏参数为初始值，此后每50期新开奖重新评估。
half_life, prior_strength, calibration_state, calibration_status = calibrate_every_50(
    df, half_life, prior_strength
)
st.info(
    f"统计参数校准：{calibration_status}｜当前半衰期 {half_life} 期｜"
    f"平滑强度 {prior_strength}｜累计完成 {calibration_state.get('completed_batches', 0)} 轮"
)
probs = ensemble_probs(df, half_life, prior_strength)

@st.cache_data(ttl=3600, show_spinner=False, max_entries=4)
def cached_dynamic_weights(frame, hl, prior):
    return dynamic_model_weights(frame, hl, prior, validation_tests=100)

# Recalculate adaptive weights only once per 50 new draws, not on each refresh.
# The next prediction still uses all latest historical draws in ensemble_probs.
weight_end = len(df) - (len(df) % 50)
weight_frame = df.iloc[:max(100, weight_end)].copy()
weight_state = cached_dynamic_weights(weight_frame, half_life, prior_strength)
dynamic_probs = apply_dynamic_weights(probs, weight_state)
probs["big"]["final"] = dynamic_probs["big"]
probs["odd"]["final"] = dynamic_probs["odd"]
save_weight_snapshot(int(weight_frame["期号"].iloc[-1]), weight_state, half_life, prior_strength)
ledger_df = update_forecast_ledger(df, probs)
p_big = probs["big"]["final"]
p_odd = probs["odd"]["final"]
big_pick = "大" if p_big >= 0.5 else "小"
big_edge = abs(p_big - 0.5)
odd_pick = "单" if p_odd >= 0.5 else "双"
odd_edge = abs(p_odd - 0.5)

# V6.1 prospective 7-of-10 forecasts, separate from the existing big/odd ledger.
# Read the existing ledger first so that calibration counts only actual prospective predictions.
# The update function also settles past entries and locks the current issue exactly once.
# First settle existing records without altering a previously locked forecast.
seven_forecast = seven_digit_forecast(df, half_life)
seven_ledger = update_seven_digit_ledger(df, seven_forecast)
digit_calibration = calibrate_digit_models(df, seven_ledger, half_life)
# Calibrated parameters apply to the NEXT issue; existing locked forecasts remain unchanged.
seven_forecast = seven_digit_forecast(df, half_life, digit_calibration)
report_status = build_digit_reports(seven_ledger)

# V6.4 six-digit models are independent of the existing seven-digit records.
# Start with current calibrated half-life, then lock the six-digit issue once.
six_default_cfg = {pos: {"half_life": half_life} for pos in ("A", "B", "C")}
six_initial = six_digit_forecast(df, six_default_cfg)
six_ledger = update_six_digit_ledger(df, six_initial)
six_calibration = calibrate_six_models(df, six_ledger, half_life)
six_forecast = six_digit_forecast(df, six_calibration)
six_report_status = build_six_reports(six_ledger)

st.subheader("V6.5 区域一：三球六码独立预测")
st.caption(f"目标：第 {latest_issue + 1} 期｜六码独立滚动验证选模；每球6个不重复数字，理论覆盖率60%。")
six_columns = st.columns(3)
for pos, column in zip(("A", "B", "C"), six_columns):
    with column:
        info = six_forecast[pos]
        stored = six_ledger[pd.to_numeric(six_ledger["目标期号"], errors="coerce") == latest_issue + 1]
        saved = str(stored.iloc[0][f"{pos}六码"]) if not stored.empty else info["six"]
        st.markdown(f"#### 第{'ABC'.index(pos)+1}球 {pos}")
        st.markdown(f"### `{saved}`")
        st.caption("首次生成即锁定；不随刷新改变")
        # The visible six digits come from the write-once ledger. Never label
        # them with a freshly recomputed model or estimated probability.
        locked = stored.iloc[0] if not stored.empty else None
        locked_model = str(locked.get(f"{pos}锁定模型", "")) if locked is not None else ""
        if locked_model in ("", "nan", "None"):
            st.caption("本期锁定模型：旧记录未保存模型名称（无法追溯；不事后补填）")
        else:
            st.caption(f"本期锁定模型：{locked_model}")
        locked_rate = pd.to_numeric(locked.get(f"{pos}锁定估计覆盖率"), errors="coerce") if locked is not None else np.nan
        if pd.notna(locked_rate):
            st.caption(f"锁定时模型估计覆盖率：{float(locked_rate)*100:.1f}%（非可靠未来命中概率）")
        st.caption(f"当前重新计算的候选模型：{info['selected_model']}｜{info['selection_reason']}（仅供下一期参考）")
        with st.expander("六码独立模型验证及数字概率（当前重算，可能不同于锁定时）"):
            st.dataframe(pd.DataFrame({"模型": list(info["validation_scores"]),
                         "滚动验证覆盖率": [f"{v*100:.1f}%" if v is not None else "—" for v in info["validation_scores"].values()]}),
                         hide_index=True, use_container_width=True)
            st.dataframe(pd.DataFrame({"数字": range(10),
                         "估计概率": [f"{v*100:.2f}%" for v in info["probs"]]}),
                         hide_index=True, use_container_width=True)
with st.expander("六码独立校准状态与历史", expanded=True):
    st.dataframe(pd.DataFrame([{"球位": pos, "已核对期数": cfg["settled"],
                  "已完成校准轮数": cfg["rounds"], "当前半衰期": cfg["half_life"],
                  "距离下一轮": cfg["remaining"]} for pos, cfg in six_calibration.items()]),
                 hide_index=True, use_container_width=True)
    if V64_SIX_HISTORY.exists():
        st.dataframe(pd.read_csv(V64_SIX_HISTORY).tail(20), hide_index=True, use_container_width=True)
with st.expander("六码事前预测记录与开奖核对", expanded=False):
    columns = ["目标期号"]
    for pos in ("A", "B", "C"):
        columns += [f"{pos}六码", f"{pos}实际号码", f"{pos}是否命中"]
    st.dataframe(six_ledger[columns].sort_values("目标期号", ascending=False).head(100),
                 hide_index=True, use_container_width=True)
    rows = []
    for pos in ("A", "B", "C"):
        settled = six_ledger[six_ledger[f"{pos}是否命中"].isin(("正确", "错误"))]
        n = len(settled)
        hits = int((settled[f"{pos}是否命中"] == "正确").sum())
        rows.append({"球位": pos, "已核对": n, "命中": hits,
                     "覆盖率": f"{hits/n*100:.1f}%" if n else "—", "理论基准": "60.0%"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.download_button("下载六码事前预测CSV", six_ledger.to_csv(index=False).encode("utf-8-sig"),
                       file_name=V64_SIX_FILE.name, mime="text/csv")
with st.expander("六码50期分档与北京时间每日整理", expanded=False):
    st.caption(f"已完成分档：{six_report_status['batches']}（三球合计）｜每日统计：{six_report_status['daily']}条")
    for path, label in ((V64_SIX_BATCHES, "六码50期分档"), (V64_SIX_DAILY, "六码每日整理")):
        if path.exists():
            st.markdown(f"**{label}**")
            st.dataframe(pd.read_csv(path).tail(30), hide_index=True, use_container_width=True)
            st.download_button(f"下载{label}", path.read_bytes(), file_name=path.name, mime="text/csv")
        else:
            st.info(f"{label}尚无符合条件的记录。")
st.divider()
st.subheader("V6.5 区域二：原有七码分析与报告")
st.caption("以下保留V6.3.1原有七码台账、独立校准、50期分档及日报；不修改历史记录。")

st.subheader("50期分档与北京时间每日整理")
st.caption("每50期已核对预测形成一档；北京时间19:05后生成当日阶段报告，次日可补全。计划休市不等于确认停盘；有新开奖仍持续同步。仅网页运行时执行，不保证无人访问时自动运行。")
st.caption(f"已完成分档：{report_status['batches']}（三球合计）｜已生成日统计：{report_status['daily']}（按日期和球位）")
with st.expander("查看分档成绩与每日整理", expanded=False):
    if V63_DIGIT_BATCHES.exists():
        st.dataframe(pd.read_csv(V63_DIGIT_BATCHES).tail(30), hide_index=True, use_container_width=True)
        st.download_button("下载50期分档成绩", V63_DIGIT_BATCHES.read_bytes(), file_name=V63_DIGIT_BATCHES.name, mime="text/csv")
    if V63_DAILY_REPORT.exists():
        st.dataframe(pd.read_csv(V63_DAILY_REPORT).tail(30), hide_index=True, use_container_width=True)
        st.download_button("下载每日整理报告", V63_DAILY_REPORT.read_bytes(), file_name=V63_DAILY_REPORT.name, mime="text/csv")

st.subheader("三球七码独立校准预测")
st.caption("每球独立累计50期已核对的真实事前预测后校准一次；当前期已锁定预测不变，参数仅影响未来新预测。")
with st.expander("三球独立校准状态与历史", expanded=True):
    st.dataframe(pd.DataFrame([{"球位": pos, "已核对期数": c["settled"],
                                "已完成校准轮数": c["rounds"],
                                "当前半衰期": c["half_life"],
                                "距离下一轮": c["remaining"]}
                               for pos, c in digit_calibration.items()]),
                 hide_index=True, use_container_width=True)
    if V63_DIGIT_HISTORY.exists():
        st.dataframe(pd.read_csv(V63_DIGIT_HISTORY).tail(20), hide_index=True, use_container_width=True)
st.caption(f"目标：第 {latest_issue + 1} 期｜A、B、C分别从0–9中选7个不重复数字，按历史估计概率降序排列。")
seven_columns = st.columns(3)
for position, column in zip(("A", "B", "C"), seven_columns):
    with column:
        info = seven_forecast[position]
        st.markdown(f"#### 第{'ABC'.index(position) + 1}球 {position}")
        # The locked pre-draw forecast takes precedence over any recomputation.
        stored = seven_ledger[pd.to_numeric(seven_ledger["目标期号"], errors="coerce") == latest_issue + 1]
        saved = str(stored.iloc[0][f"{position}七码"]) if not stored.empty else info["seven"]
        st.markdown(f"### `{saved}`")
        st.caption("本期已锁定的七码（首次预测，按当时概率排序）")
        st.caption(f"当前模型估计7码覆盖概率：{info['coverage_estimate']*100:.1f}%（仅供统计研究，不是可靠的真实命中概率）")
        with st.expander("五模型权重与时间顺序验证"):
            st.caption(f"最近100期内每隔5期进行一次事前验证，共{info['validation_n']}个验证目标；模型权重经过强收缩。")
            st.dataframe(pd.DataFrame({"模型": list(info["weights"]),
                                       "权重": [f"{v*100:.1f}%" for v in info["weights"].values()]}),
                         hide_index=True, use_container_width=True)
        with st.expander("查看0–9完整概率"):
            digit_probs = info["probs"]
            st.dataframe(pd.DataFrame({"数字": list(range(10)),
                                       "估计概率": [f"{v * 100:.2f}%" for v in digit_probs]}),
                         hide_index=True, use_container_width=True)
st.caption("五模型融合：全历史、近500期、近100期、时间衰减、状态转移；仅用此前开奖进行验证，权重收缩并向均匀分布混合。公平独立开奖时单球7码理论覆盖率仍为70%，不能保证优于随机选择。")
with st.expander("V6.3 三球七码事前预测记录与开奖核对（继承V6.1原始日志）", expanded=False):
    st.caption("仅记录系统运行时真正生成的未来期预测；每期首次七码和概率锁定，不事后补造。")
    if seven_ledger.empty:
        st.info("尚无三球七码预测记录。")
    else:
        display_cols = ["目标期号"]
        for position in ("A", "B", "C"):
            display_cols += [f"{position}七码", f"{position}实际号码", f"{position}是否命中"]
        st.dataframe(seven_ledger[display_cols].sort_values("目标期号", ascending=False).head(100),
                     hide_index=True, use_container_width=True)
        score_rows = []
        for position in ("A", "B", "C"):
            settled = seven_ledger[seven_ledger[f"{position}是否命中"].isin(["正确", "错误"])]
            n = len(settled)
            score_rows.append({"球位": position, "已核对期数": n,
                               "七码命中": int((settled[f"{position}是否命中"] == "正确").sum()),
                               "覆盖率": f"{(settled[f'{position}是否命中'] == '正确').mean() * 100:.1f}%" if n else "—",
                               "随机基准": "70.0%"})
        st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)
        st.download_button("下载V6.1三球七码事前预测CSV",
                           data=seven_ledger.to_csv(index=False).encode("utf-8-sig"),
                           file_name="v61_seven_digit_log.csv", mime="text/csv")

st.subheader("最近50期大小单双走势分析")
recent_50 = df.tail(50).copy()
trend_big = trend_50_stats((recent_50["和值"].to_numpy() >= 14).astype(int))
trend_odd = trend_50_stats((recent_50["和值"].to_numpy() % 2 == 1).astype(int))
t1, t2 = st.columns(2)
with t1:
    st.markdown("#### 大小走势")
    st.write(f"最近{trend_big['n']}期：大 {trend_big['yes']} 期 / 小 {trend_big['no']} 期")
    st.progress(trend_big['yes'] / max(1, trend_big['n']))
    st.caption(f"当前连续{'大' if trend_big['last'] else '小'} {trend_big['streak']} 期｜大小转换 {trend_big['changes']} 次")
    tb = probs['big']['trend50']
    st.write(f"**50期走势模型：{'大' if tb >= .5 else '小'}**｜大 {tb*100:.1f}% / 小 {(1-tb)*100:.1f}%")
with t2:
    st.markdown("#### 单双走势")
    st.write(f"最近{trend_odd['n']}期：单 {trend_odd['yes']} 期 / 双 {trend_odd['no']} 期")
    st.progress(trend_odd['yes'] / max(1, trend_odd['n']))
    st.caption(f"当前连续{'单' if trend_odd['last'] else '双'} {trend_odd['streak']} 期｜单双转换 {trend_odd['changes']} 次")
    to = probs['odd']['trend50']
    st.write(f"**50期走势模型：{'单' if to >= .5 else '双'}**｜单 {to*100:.1f}% / 双 {(1-to)*100:.1f}%")
with st.expander("最近50期开奖记录", expanded=False):
    st.dataframe(recent_50[["期号", "开奖时间", "A", "B", "C", "和值", "大小", "单双"]].iloc[::-1],
                 use_container_width=True, hide_index=True)
st.caption("走势模型采用频率、短期变化和状态转换，并向50%收缩；它是统计估计，不意味着随机开奖存在可预测规律。")
st.divider()

with st.expander("V5.9 九模型动态权重与自动校准", expanded=True):
    st.write(
        f"大小权重验证样本：{weight_state.get('validation_n', 0)}期｜"
        f"单双权重验证样本：{weight_state.get('validation_n', 0)}期"
    )
    weight_rows = []
    for key, name in MODEL_NAMES.items():
        weight_rows.append({
            "模型": name,
            "大小权重": f"{weight_state['big'][key] * 100:.1f}%",
            "单双权重": f"{weight_state['odd'][key] * 100:.1f}%",
        })
    st.dataframe(pd.DataFrame(weight_rows), use_container_width=True, hide_index=True)
    st.caption(
        "权重每次新数据触发重算，每50期重算一次；每50期仍执行原统计参数校准。"
        "权重由最近100个可用目标的逐期回测命中率和Brier分数估计，并向等权重收缩，"
        "以降低短期过拟合。滚动验证仍可能有模型选择偏差，不保证未来命中率提高。"
    )

st.subheader("下一期预测")
latest_time_beijing = format_beijing_time(latest_time)
target_issue = latest_issue + 1
st.caption(
    f"当前数据截至：第 {latest_issue} 期"
    + (f"｜最近开奖时间（北京时间）：{latest_time_beijing}" if latest_time_beijing else "")
    + f"｜下一期目标：第 {target_issue} 期｜有效历史：{len(df):,} 期"
)

st.markdown("### 大小预测")
st.metric("大（14–27）", f"{p_big * 100:.1f}%")
st.progress(float(np.clip(p_big, 0, 1)))
st.metric("小（0–13）", f"{(1-p_big) * 100:.1f}%")
st.markdown(f"**分析结果：{big_pick}**")
st.caption(f"与50%基准的概率差：{big_edge*100:.1f}个百分点")

st.divider()

st.markdown("### 单双预测")
st.metric("单", f"{p_odd * 100:.1f}%")
st.progress(float(np.clip(p_odd, 0, 1)))
st.metric("双", f"{(1-p_odd) * 100:.1f}%")
st.markdown(f"**分析结果：{odd_pick}**")
st.caption(f"与50%基准的概率差：{odd_edge*100:.1f}个百分点")

st.info(
    "方案B：每期都给出大小和单双方向，哪怕概率接近50%。"
    "这些概率是模型估计，不保证未来命中率高于50%。"
)

# 结果和值拆解：例如 A=4、B=4、C=8，则显示 4 + 4 + 8 = 16。
st.markdown("### 最近一期开奖结果")
latest_row = df.sort_values("期号").iloc[-1]
a_num, b_num, c_num = int(latest_row["A"]), int(latest_row["B"]), int(latest_row["C"])
draw_sum = int(latest_row["和值"])
r1, r2, r3, r4 = st.columns(4)
r1.metric("开奖号码", f"{a_num} + {b_num} + {c_num}")
r2.metric("和值", str(draw_sum))
r3.metric("大小", str(latest_row["大小"]))
r4.metric("单双", str(latest_row["单双"]))

# 直接读取 pc28.world /draws/latest 返回的 countdown 字段，不再用开奖时间+210秒估算。
st.markdown("### 下一期开奖倒计时")

@st.fragment(run_every=5)
def draw_countdown():
    try:
        latest_payload = fetch_latest_payload()
        countdown_data = latest_payload.get("countdown", {})
        if not isinstance(countdown_data, dict):
            countdown_data = {}

        seconds_value = countdown_data.get("seconds")
        next_draw_at = countdown_data.get("next_draw_at")
        fetched_at = beijing_now()

        if next_draw_at is not None:
            # API 的 next_draw_at 使用 Unix 毫秒时间戳。
            target_timestamp = float(next_draw_at)
            if target_timestamp > 10_000_000_000:
                target_timestamp /= 1000.0
            target_dt = datetime.fromtimestamp(target_timestamp, tz=BEIJING_TZ)
            seconds_left = max(0, int((target_dt - fetched_at).total_seconds()))
            time_text = target_dt.strftime("%Y-%m-%d %H:%M:%S")
            source_label = "下一期开奖时间来自 pc28.world 实时接口"
        elif seconds_value is not None:
            # 兼容只返回 seconds 的接口版本，以服务器返回的秒数为基准。
            seconds_left = max(0, int(float(seconds_value)))
            target_dt = fetched_at + timedelta(seconds=seconds_left)
            time_text = target_dt.strftime("%Y-%m-%d %H:%M:%S")
            source_label = "倒计时秒数来自 pc28.world 实时接口"
        else:
            st.warning("数据源暂未返回 countdown.seconds 或 countdown.next_draw_at，无法显示实时倒计时。")
            st.caption("系统不会再用210秒间隔自行估算。请稍后重试。")
            return

        mins, secs = divmod(seconds_left, 60)
        hours, mins = divmod(mins, 60)
        countdown_text = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"
        if seconds_left > 0:
            st.metric("实时预计开奖倒计时", countdown_text)
        else:
            st.warning("数据源倒计时已到，等待最新开奖结果同步。")
        st.caption(f"预计开奖时间（北京时间）：{time_text}｜{source_label}｜每5秒刷新一次")
    except Exception as exc:
        st.warning(f"暂时无法读取数据源实时倒计时：{exc}")
        st.caption("请稍后刷新页面；自动同步历史开奖仍保持每210秒一次。")

draw_countdown()

with st.expander("查看模型分项概率", expanded=False):
    detail_rows = []
    for target, title, labels in [
        ("big", "大小", ("大", "小")),
        ("odd", "单双", ("单", "双")),
    ]:
        result = probs[target]
        for key, name in [*MODEL_NAMES.items(), ("final", "动态权重综合")]:
            p = result[key]
            detail_rows.append({
                "玩法": title,
                "统计模型": name,
                labels[0]: f"{p*100:.1f}%",
                labels[1]: f"{(1-p)*100:.1f}%",
            })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

with st.expander("V6.0 九模型真实事前分段对比（50 / 100 / 200 / 500期）", expanded=True):
    st.caption("仅使用V5.9以来真实开奖前保存的九模型概率；所有模型使用同一组已核对期号。"
               "不同窗口相互重叠，不能当成独立实验；历史排名不代表未来优势。")
    selected_window = st.selectbox("查看最近多少期真实事前记录", [50, 100, 200, 500],
                                   index=2, key="v60_prospective_window")
    for target, label in (("big", "大小"), ("odd", "单双")):
        st.markdown(f"#### {label}九模型分段成绩")
        window_table = prospective_model_window_scores(ledger_df, target, selected_window)
        if window_table.empty:
            st.info("暂无完整九模型事前记录。")
            continue
        display = window_table.copy()
        display["命中率"] = display["命中率"].map(
            lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%")
        display["Brier"] = display["Brier"].map(lambda x: f"{x:.5f}")
        display["相对50%基准"] = display["相对50%基准"].map(lambda x: f"{x:+.5f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(f"本窗口共同有效样本：{int(window_table['已核对期数'].iloc[0])}期。"
                   "Brier低于0.25000表示样本内较优，不代表统计显著或可预测未来。")

with st.expander("V6.0 动态权重历史记录", expanded=False):
    st.caption("只从V6.0启用后开始记录权重快照，不补造以前的权重。"
               "此表用于核对权重变化，不自动根据短期第一名激进调权。")
    if V60_WEIGHT_HISTORY_FILE.exists():
        try:
            weight_history = pd.read_csv(V60_WEIGHT_HISTORY_FILE)
            st.write(f"已保存 {len(weight_history)} 条权重快照（大小、单双分别记录）。")
            st.dataframe(weight_history.tail(30).iloc[::-1],
                         use_container_width=True, hide_index=True)
            st.download_button("下载V6.0权重历史CSV",
                data=weight_history.to_csv(index=False).encode("utf-8-sig"),
                file_name="v60_weight_history.csv", mime="text/csv")
        except Exception as exc:
            st.warning(f"暂时无法读取权重历史：{type(exc).__name__}")
    else:
        st.info("等待首次权重快照。")

with st.expander("九模型完整历史回测（逐期向前验证）", expanded=False):
    test_count = st.selectbox("回测最近多少期", [100, 300, 500], index=0)
    if st.button("运行九模型与动态权重回测"):
        with st.spinner("正在运行低CPU回测；机器学习每150个测试目标重新训练，请稍候…"):
            result = rolling_backtest(df, half_life, prior_strength, tests=test_count)
        if result:
            st.write(f"测试期数：{result['n']}｜权重预热期数：{result.get('warmup', 0)}")
            for note in result.get("notes", []):
                st.warning(note)
            display = result["models"].copy()
            for col in ["大小命中率", "单双命中率"]:
                display[col] = display[col].map(lambda x: f"{x*100:.1f}%")
            for col in ["大小Brier分数", "单双Brier分数"]:
                display[col] = display[col].map(lambda x: f"{x:.4f}")
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption("新增『动态权重综合（严格逐期）』和『50%固定基准』。动态权重只使用当期以前已核对的模型成绩，最多回看300期；与九模型等权综合并排比较。机器学习每150个目标重训一次；线上每次计算新一期概率仍会训练轻量模型，回测与线上实现并非完全相同。历史成绩不能保证未来命中率。")
        else:
            st.warning("有效历史数据不足，至少需要120期。")

with st.expander("V5.9 五段验证与一致性核对（500期，每段100期）", expanded=False):
    st.caption("同一次逐期回测中按时间先后划分五段；每段100期。每一期只使用之前的数据，模型权重也仅依据已开奖历史。")
    if st.button("运行500期分段验证", key="run_v56_segments"):
        with st.spinner("正在运行500期分段验证，机器学习每150个目标重训一次，请勿重复点击…"):
            segment_result = rolling_backtest(df, half_life, prior_strength, tests=500)
        if segment_result is None or segment_result["n"] < 500:
            st.warning("有效数据不足，无法完成完整500期分段验证。")
        else:
            records = segment_result["draw_results"].copy()
            records["段号"] = (records["index"] // 100 + 1).astype(int)
            model_labels = {"pattern": "历史相似状态", "final": "九模型等权综合",
                            "dynamic": "动态权重综合", "baseline": "50%固定基准"}
            records["模型"] = records["model"].map(model_labels)
            records["项目"] = records["target"].map({"big": "大小", "odd": "单双"})
            summary = (records.groupby(["段号", "项目", "模型"], as_index=False)
                       .agg(期数=("hit", "size"), 命中率=("hit", "mean"), Brier分数=("brier", "mean")))
            summary["时间段"] = summary["段号"].map(lambda x: f"第{x}段（从旧到新）")
            summary["命中率"] = summary["命中率"].map(lambda x: f"{x*100:.1f}%")
            summary["Brier分数"] = summary["Brier分数"].map(lambda x: f"{x:.4f}")
            st.dataframe(summary[["时间段", "项目", "模型", "期数", "命中率", "Brier分数"]],
                         use_container_width=True, hide_index=True)
            st.download_button("下载分段验证CSV", data=summary.to_csv(index=False).encode("utf-8-sig"),
                               file_name="canada28_v56_segments.csv", mime="text/csv")
            # Compare the exact same underlying draw-level records, never independently rerun.
            st.markdown("#### 五段与500期汇总一致性检查")
            checks = []
            full_table = segment_result["models"].set_index("模型")
            for key, model_name in model_labels.items():
                for target, label, hit_col, brier_col in (
                    ("big", "大小", "大小命中率", "大小Brier分数"),
                    ("odd", "单双", "单双命中率", "单双Brier分数"),
                ):
                    subset = records[(records["model"] == key) & (records["target"] == target)]
                    # records["model"] retains the original key, even though the display label is mapped.
                    n = len(subset)
                    hits = int(subset["hit"].sum())
                    total_hit = float(full_table.loc[model_name if key != "dynamic" else
                        "动态权重综合（严格逐期）", hit_col])
                    total_brier = float(full_table.loc[model_name if key != "dynamic" else
                        "动态权重综合（严格逐期）", brier_col])
                    ok = (n == 500 and abs(hits / n - total_hit) < 1e-10
                          and abs(float(subset["brier"].mean()) - total_brier) < 1e-10)
                    checks.append({"项目": label, "模型": model_name, "分段命中数": hits,
                                   "500期汇总命中数": round(total_hit * 500),
                                   "检查": "一致" if ok else "不一致"})
            check_df = pd.DataFrame(checks)
            st.dataframe(check_df, use_container_width=True, hide_index=True)
            if (check_df["检查"] == "一致").all():
                st.success("同一次回测的分段结果与500期汇总完全一致。")
            else:
                st.error("检测到分段与汇总不一致，请下载结果检查。")
            st.caption("不同时间运行的两次回测可能因新开奖进入500期窗口而不同；应以同一次计算的核对结果为准。")
            st.caption("分段结果只代表过去500期，不能证明未来可预测；固定50%概率的方向命中率受平局判定规则影响，主要比较Brier分数。")

with st.expander("逐期预测记录与开奖核对", expanded=True):
    st.write(f"累计预测记录：{len(ledger_df)} 期")
    st.caption("仅记录系统实际运行时生成的事前预测；离线或接口中断期间的期数不会事后补造预测。")

    st.markdown("#### 大小预测记录")
    big_table = ledger_df[[
        "目标期号", "实际大小", "预测大小", "大小结果"
    ]].copy()
    big_table = big_table.rename(columns={
        "目标期号": "期号", "实际大小": "实际大小", "预测大小": "预测大小", "大小结果": "结果"
    })
    big_table["期号"] = pd.to_numeric(big_table["期号"], errors="coerce")
    big_table = big_table.sort_values("期号", ascending=False).head(50)
    big_table["结果"] = big_table["结果"].map(
        lambda x: {"正确": "✅ 正确", "错误": "❌ 错误", "等待开奖": "⏳ 等待开奖"}.get(str(x), str(x)))
    big_table["实际大小"] = big_table["实际大小"].fillna("待开奖").replace({"None": "待开奖"})
    st.dataframe(big_table, use_container_width=True, hide_index=True)

    st.markdown("#### 单双预测记录")
    odd_table = ledger_df[[
        "目标期号", "实际单双", "预测单双", "单双结果"
    ]].copy()
    odd_table = odd_table.rename(columns={
        "目标期号": "期号", "实际单双": "实际单双", "预测单双": "预测单双", "单双结果": "结果"
    })
    odd_table["期号"] = pd.to_numeric(odd_table["期号"], errors="coerce")
    odd_table = odd_table.sort_values("期号", ascending=False).head(50)
    odd_table["结果"] = odd_table["结果"].map(
        lambda x: {"正确": "✅ 正确", "错误": "❌ 错误", "等待开奖": "⏳ 等待开奖"}.get(str(x), str(x)))
    odd_table["实际单双"] = odd_table["实际单双"].fillna("待开奖").replace({"None": "待开奖"})
    st.dataframe(odd_table, use_container_width=True, hide_index=True)

    resolved_big = ledger_df[ledger_df["实际大小"].astype(str).isin(["大", "小"])]
    resolved_odd = ledger_df[ledger_df["实际单双"].astype(str).isin(["单", "双"])]
    st.caption(
        f"大小已核对 {len(resolved_big)} 期，命中 {(resolved_big['大小结果'] == '正确').sum()} 期；"
        f"单双已核对 {len(resolved_odd)} 期，命中 {(resolved_odd['单双结果'] == '正确').sum()} 期。"
    )
    st.download_button("下载逐期预测记录 CSV",
        data=ledger_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="canada28_forecast_log.csv", mime="text/csv")
    st.caption("每个目标期只保留首次预测概率，开奖后按期号分别核对。Streamlit 云端本地文件可能因重启丢失，请定期下载备份。")

with st.expander("V5.9 未来开奖独立验证（仅事前记录）", expanded=True):
    st.caption("只统计系统实际运行时提前保存、并在开奖后核对的记录；历史500期回测不计入此处。")
    verified = ledger_df.copy()
    for target, label, actual_col, pick_col, p_col, shadow_pick, shadow_p in (
        ("big", "大小", "实际大小", "预测大小", "大小概率大",
         "相似模型大小预测", "相似模型大小概率大"),
        ("odd", "单双", "实际单双", "预测单双", "单双概率单",
         "相似模型单双预测", "相似模型单双概率单"),
    ):
        st.markdown(f"#### {label}独立验证")
        sample = verified[
            verified[actual_col].astype(str).isin(["大", "小"] if target == "big" else ["单", "双"])
            & verified[shadow_pick].astype(str).isin(["大", "小"] if target == "big" else ["单", "双"])
        ].copy()
        if sample.empty:
            st.info(f"{label}暂无同时具备两种模型事前预测的已开奖记录；等待未来开奖积累样本。")
            continue
        y = (sample[actual_col].astype(str) == ("大" if target == "big" else "单")).astype(int)
        rows = []
        for name, pick, p in (
            ("当前动态权重", pick_col, p_col),
            ("历史相似状态（影子模型）", shadow_pick, shadow_p),
        ):
            prob = pd.to_numeric(sample[p], errors="coerce")
            valid = prob.notna() & prob.between(0, 1) & sample[pick].notna()
            if valid.sum() == 0:
                continue
            rows.append({"模型": name, "有效期数": int(valid.sum()),
                         "命中率": f"{(sample.loc[valid, pick].astype(str) == sample.loc[valid, actual_col].astype(str)).mean()*100:.1f}%",
                         "Brier分数": f"{((prob[valid] - y[valid])**2).mean():.4f}"})
        rows.append({"模型": "固定50%概率", "有效期数": len(sample),
                     "命中率": "—（不指定方向）", "Brier分数": "0.2500"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"当前样本 {len(sample)} 期；少量样本的波动很大，不应作为投注依据。")
    st.download_button("下载V5.9事前预测核对记录CSV",
                       data=verified.to_csv(index=False).encode("utf-8-sig"),
                       file_name="canada28_v59_prospective_log.csv", mime="text/csv")
    if get_database_url():
        st.info("已配置持久PostgreSQL数据库，仍建议定期下载CSV备份。")
    else:
        st.warning("未配置持久数据库，云端本地文件可能在重启后丢失，请下载CSV备份。")

with st.expander("V5.9 九模型真实事前独立成绩", expanded=True):
    st.caption("只统计升级后在开奖前保存的各模型概率。旧预测不回填，确保不把历史回测冒充事前预测。")
    for target, label, actual_col, positive in (
        ("big", "大小", "实际大小", "大"),
        ("odd", "单双", "实际单双", "单"),
    ):
        st.markdown(f"#### {label}：九模型独立成绩")
        records = ledger_df[ledger_df[actual_col].astype(str).isin(
            ["大", "小"] if target == "big" else ["单", "双"]
        )].copy()
        y_all = (records[actual_col].astype(str) == positive).astype(int)
        rows = []
        for key, name in MODEL_NAMES.items():
            p = pd.to_numeric(records[f"V59_{target}_{key}_p"], errors="coerce")
            valid = p.notna() & p.between(0, 1)
            if not valid.any():
                continue
            y = y_all[valid]
            pp = p[valid]
            rows.append({"模型": name, "已核对期数": int(valid.sum()),
                         "命中率": float(((pp >= 0.5).astype(int) == y).mean()),
                         "Brier": float(((pp-y)**2).mean())})
        if not rows:
            st.info("尚无V5.9九模型事前记录；等待新一期开奖后自动积累。")
        else:
            table = pd.DataFrame(rows).sort_values(["Brier", "命中率"], ascending=[True, False])
            table["命中率"] = table["命中率"].map(lambda v: f"{v*100:.1f}%")
            table["Brier"] = table["Brier"].map(lambda v: f"{v:.5f}")
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.caption("固定50%概率基准Brier=0.25000；短期排名不稳定，不能保证提高未来胜率。")
    st.download_button("下载九模型真实事前预测CSV",
                       data=ledger_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="canada28_v59_nine_models.csv", mime="text/csv")

with st.expander("V5.9 动态权重诊断", expanded=False):
    st.caption("权重来自最近100期逐期回测并向等权重收缩；历史第一名不等于未来第一名。")
    for target, label, hit_col, brier_col in (
        ("big", "大小", "大小命中率", "大小Brier分数"),
        ("odd", "单双", "单双命中率", "单双Brier分数"),
    ):
        st.markdown(f"#### {label}权重")
        table = weight_state.get("table", pd.DataFrame())
        if table is None or table.empty:
            st.info("暂无可用验证数据。")
            continue
        table = table.set_index("模型")
        rows = []
        for key, name in MODEL_NAMES.items():
            if name in table.index:
                rows.append({"模型": name,
                             "验证命中率": f"{float(table.loc[name, hit_col])*100:.1f}%",
                             "验证Brier": f"{float(table.loc[name, brier_col]):.5f}",
                             "当前权重": f"{weight_state[target][key]*100:.2f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("这里的权重验证窗口与手动回测窗口可能不同。")

with st.expander("V5.9 真实事前成绩：50 / 100 / 500期", expanded=True):
    score_rows = []
    for label, actual_col, prediction_col, probability_col, positive in (
        ("大小", "实际大小", "预测大小", "大小概率大", "大"),
        ("单双", "实际单双", "预测单双", "单双概率单", "单"),
    ):
        settled = ledger_df[ledger_df[actual_col].astype(str).isin(
            ["大", "小"] if label == "大小" else ["单", "双"]
        )].copy()
        settled["目标期号"] = pd.to_numeric(settled["目标期号"], errors="coerce")
        settled = settled.sort_values("目标期号")
        for window in (50, 100, 500):
            sample = settled.tail(window)
            p = pd.to_numeric(sample[probability_col], errors="coerce")
            valid = p.notna() & p.between(0, 1)
            sample, p = sample.loc[valid], p.loc[valid]
            n = len(sample)
            if not n:
                score_rows.append({"项目": label, "窗口": f"最近{window}期",
                                   "已核对": 0, "命中率": "—", "Brier分数": "—"})
                continue
            hits = (sample[prediction_col].astype(str) == sample[actual_col].astype(str)).sum()
            y = (sample[actual_col].astype(str) == positive).astype(int)
            score_rows.append({"项目": label, "窗口": f"最近{window}期",
                               "已核对": n, "命中率": f"{hits/n*100:.1f}%",
                               "Brier分数": f"{((p-y)**2).mean():.4f}"})
    st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
    st.caption("窗口不足50/100/500期时显示实际样本数；只统计真实保存并已核对的事前预测，不补造历史。")

with st.expander("V5.9 预测台账一致性与漏期检查", expanded=True):
    anomalies, issues = ledger_audit(ledger_df, df)
    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("未发现重复目标、目标期号错位、台账间断或实际开奖冲突。")
    st.caption("期号跳跃可能是网站未运行时造成的，并不代表系统伪造预测。")
    if not anomalies.empty:
        st.dataframe(anomalies, use_container_width=True, hide_index=True)

with st.expander("每50期开奖校准历史", expanded=False):
    if CALIBRATION_LOG_FILE.exists():
        try:
            calibration_history = pd.read_csv(CALIBRATION_LOG_FILE)
            st.dataframe(calibration_history.iloc[::-1], use_container_width=True, hide_index=True)
            st.download_button(
                "下载模型校准历史 CSV",
                data=calibration_history.to_csv(index=False).encode("utf-8-sig"),
                file_name="canada28_calibration_history.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.warning(f"暂时无法读取校准历史：{exc}")
    else:
        st.caption("尚未完成第一轮50期开奖校准。")

with st.expander("最近开奖记录", expanded=False):
    recent_draws = df.tail(30).iloc[::-1].copy()
    recent_draws["和值拆解"] = recent_draws.apply(
        lambda row: f"{int(row['A'])} + {int(row['B'])} + {int(row['C'])} = {int(row['和值'])}",
        axis=1,
    )
    st.dataframe(recent_draws, use_container_width=True, hide_index=True)

csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "下载历史数据 CSV 备份",
    data=csv_bytes,
    file_name="canada28_history_backup.csv",
    mime="text/csv",
)

st.caption(
    "数据来自 pc28.world，使用时请注明并链接 https://pc28.world。"
    "本应用仅作数据统计与研究，不提供投注功能；每50期仅校准统计平滑参数，大小与单双分别采用九模型动态权重，不保证未来命中率提高。"
)
st.caption(
    "重要：Streamlit Cloud 本地文件可能在重启或重新部署后丢失；"
    "请定期下载 CSV 备份。如已配置 PostgreSQL 则使用云数据库；否则本地存储不保证持久。"
)
