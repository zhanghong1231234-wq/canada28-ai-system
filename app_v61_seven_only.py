# -*- coding: utf-8 -*-
"""Canada28 Matrix AI V6.1 · 三算法七码专版
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
                     "v60_weight_history.csv", "v61_seven_digit_log.csv"):
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
    page_title="Canada28 Matrix AI V6.1 · 三算法七码专版",
    page_icon="📊",
    layout="wide",
)

st.title("Canada28 Matrix AI V6.1 · 三算法七码专版")
st.caption(
    "A、B、C 三球七码统计研究（全历史 / 最近100期 / 时间加权）。数据来自 pc28.world；"
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
SEVEN_ALGORITHMS = {
    "all": "全历史频率",
    "recent100": "最近100期频率",
    "decay": "时间加权频率",
}

V61_DIGIT_FILE = DATA_DIR / "v61_seven_digit_log.csv"
V61_DIGIT_COLS = ["预测时间", "目标期号", "基于最新期号", "七码算法版本"]
for _position in ("A", "B", "C"):
    V61_DIGIT_COLS += ([f"{_position}七码", f"{_position}实际号码", f"{_position}是否命中"]
                       + [f"{_position}算法_{key}" for key in SEVEN_ALGORITHMS]
                       + [f"{_position}概率{digit}" for digit in range(10)])



def seven_digit_probabilities(history, position, half_life=500):
    """仅使用该球位已经开奖的历史，三个简单频率估计等权融合。"""
    values = pd.to_numeric(history[position], errors="coerce").dropna().astype(int).to_numpy()
    values = values[(values >= 0) & (values <= 9)]

    def smooth(sample, strength=50):
        counts = np.bincount(sample, minlength=10).astype(float)
        return (counts + strength / 10.0) / (len(sample) + strength)

    if not len(values):
        uniform = np.full(10, 0.1)
        return {"all": uniform.copy(), "recent100": uniform.copy(),
                "decay": uniform.copy(), "final": uniform.copy()}

    overall = smooth(values)
    recent100 = smooth(values[-100:])
    age = np.arange(len(values) - 1, -1, -1, dtype=float)
    weights = np.power(0.5, age / max(float(half_life), 1.0))
    weighted_counts = np.bincount(values, weights=weights, minlength=10)
    decayed = (weighted_counts + 5.0) / (weights.sum() + 50.0)
    combined = (overall + recent100 + decayed) / 3.0
    return {"all": overall, "recent100": recent100,
            "decay": decayed, "final": combined / combined.sum()}


def _seven_from_probabilities(probabilities):
    ranking = sorted(range(10), key=lambda digit: (-float(probabilities[digit]), digit))
    return "".join(map(str, ranking[:7]))


def seven_digit_forecast(history, half_life=500):
    result = {}
    for position in ("A", "B", "C"):
        estimates = seven_digit_probabilities(history, position, half_life)
        result[position] = {
            "probs": estimates["final"],
            "seven": _seven_from_probabilities(estimates["final"]),
            "algorithms": {key: _seven_from_probabilities(estimates[key])
                           for key in SEVEN_ALGORITHMS},
            "algorithm_probs": estimates,
        }
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
                  "目标期号": target, "基于最新期号": latest,
                  "七码算法版本": "三算法等权"}
        for position in ("A", "B", "C"):
            record[f"{position}七码"] = prediction[position]["seven"]
            for key in SEVEN_ALGORITHMS:
                record[f"{position}算法_{key}"] = prediction[position]["algorithms"][key]
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
    auto_sync = st.checkbox("开启自动同步（每210秒）", value=True)
    st.caption("网页运行时每5秒检查最新开奖，每210秒补充历史；只显示三球七码分析。")
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

# V6.1 prospective 7-of-10 forecasts, separate from the existing big/odd ledger.
seven_forecast = seven_digit_forecast(df, half_life)
seven_ledger = update_seven_digit_ledger(df, seven_forecast)

st.subheader("V6.1 三种简单算法 · 下一期三球七码统计分析")
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
        st.caption("三算法等权综合：本期首次预测锁定；旧记录仍保留原先算法，不事后重算。")
        algorithm_rows = []
        for key, label in SEVEN_ALGORITHMS.items():
            field = f"{position}算法_{key}"
            locked = str(stored.iloc[0].get(field, "")) if not stored.empty else ""
            if locked in ("", "nan", "None"):
                shown = "旧期未保存" if not stored.empty else info["algorithms"][key]
            else:
                shown = locked
            algorithm_rows.append({"算法": label, "七码": shown})
        st.dataframe(pd.DataFrame(algorithm_rows), hide_index=True, use_container_width=True)
        with st.expander("查看0–9完整概率"):
            probs = info["probs"]
            st.dataframe(pd.DataFrame({"数字": list(range(10)),
                                       "估计概率": [f"{v * 100:.2f}%" for v in probs]}),
                         hide_index=True, use_container_width=True)
st.caption("三种算法分别为全历史频率、最近100期频率、时间加权频率；三者等权融合得到综合七码。公平独立开奖下，单球7码理论覆盖率为70%，历史频率不能证明未来优于随机选择。")
with st.expander("V6.1 三球七码事前预测记录与开奖核对", expanded=False):
    st.caption("仅记录系统运行时真正生成的未来期预测；每期首次七码、三算法输出及综合概率锁定。旧期不补造算法明细。")
    if seven_ledger.empty:
        st.info("尚无三球七码预测记录。")
    else:
        display_cols = ["目标期号", "七码算法版本"]
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

st.divider()
st.subheader("最新开奖结果")
latest = df.iloc[-1]
st.write(f"第 {latest_issue} 期｜北京时间：{format_beijing_time(latest_time)}")
st.markdown(f"### {int(latest['A'])} + {int(latest['B'])} + {int(latest['C'])} = {int(latest['和值'])}")
with st.expander("最近30期开奖结果", expanded=False):
    recent_draws = df.tail(30).iloc[::-1][["期号", "开奖时间", "A", "B", "C", "和值"]].copy()
    st.dataframe(recent_draws, hide_index=True, use_container_width=True)
st.download_button("下载历史开奖CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                   file_name="canada28_history_backup.csv", mime="text/csv")
st.caption("仅提供历史统计参考；公平独立开奖下每球7码理论覆盖率为70%，不能保证未来命中率。")
