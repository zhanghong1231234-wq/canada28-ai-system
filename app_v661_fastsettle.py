# -*- coding: utf-8 -*-
"""Canada28 Matrix AI V6.6.1
三球六码与七码独立分析工具。仅作统计研究，不保证预测准确。
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
    page_title="Canada28 Matrix AI V6.6.1",
    page_icon="📊",
    layout="wide",
)
import hmac

def check_password():
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔐 Canada28 Matrix AI V6.6.1")
    st.subheader("请输入登录密码")

    password = st.text_input(
        "访问密码",
        type="password",
        key="login_password"
    )

    if st.button("登录", type="primary"):
        expected = str(st.secrets.get("APP_PASSWORD", ""))

        if expected and hmac.compare_digest(password, expected):
            st.session_state["authenticated"] = True
            st.session_state.pop("login_password", None)
            st.rerun()
        else:
            st.error("密码错误，请重新输入")

    return False


if not check_password():
    st.stop()
st.title("Canada28 Matrix AI V6.6.1｜三球六码 + 三球七码")
st.caption(
    "历史开奖数据驱动的三球六码 / 七码统计研究。数据来自 pc28.world；"
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
    """Causal 7-of-10 model selection with independent confirmation.

    Selection and confirmation are both prequential (no target leakage).
    The historical split is a screening heuristic, not proof of future advantage.
    """
    values = pd.to_numeric(history[position], errors="coerce").dropna().astype(int).to_numpy()
    values = values[(values >= 0) & (values <= 9)]
    names = ("全历史", "近500期", "近100期", "时间衰减", "状态转移")
    n = len(values)
    models = _digit_model_distributions(values, half_life)
    baseline = np.mean([models[name] for name in names], axis=0)
    baseline = .8 * baseline + .2 * .1
    chosen = None
    validation_n = 0

    # 240 recent outcomes, evaluated at stride 3 to reduce cloud computation.
    # The first half selects candidates; the second half confirms on unseen targets.
    if n >= 400:
        indices = list(range(max(150, n - 240), n, 3))
        midpoint = len(indices) // 2
        scores = {name: [[], []] for name in (*names, "等权融合")}
        for j, i in enumerate(indices):
            earlier = _digit_model_distributions(values[:i], half_life)
            blended = .8 * np.mean([earlier[name] for name in names], axis=0) + .2 * .1
            candidates = dict(earlier)
            candidates["等权融合"] = blended
            actual = int(values[i])
            for name, prob in candidates.items():
                top7 = np.argsort(-prob, kind="stable")[:7]
                scores[name][0 if j < midpoint else 1].append(int(actual in top7))
        validation_n = len(indices)
        # Select using only the first half. The confirmation half cannot
        # be used to pick among multiple candidates (avoids double dipping).
        selected = max(names, key=lambda name: (np.mean(scores[name][0]), -names.index(name)))
        selection_edge = np.mean(scores[selected][0]) - np.mean(scores["等权融合"][0])
        confirmation_edge = np.mean(scores[selected][1]) - np.mean(scores["等权融合"][1])
        confirmation_rate = np.mean(scores[selected][1])
        if selection_edge >= .025 and confirmation_edge >= .025 and confirmation_rate >= .725:
            chosen = selected

    if chosen is None:
        estimate = baseline
        weights = {name: 1.0 / len(names) for name in names}
    else:
        # Shrink selected model toward the precommitted baseline to limit variance.
        estimate = .5 * baseline + .5 * (.8 * models[chosen] + .2 * .1)
        weights = {name: .5 / len(names) + (.5 if name == chosen else 0.0) for name in names}
    estimate = np.maximum(estimate, 0.0)
    estimate /= estimate.sum()
    return estimate, weights, validation_n


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
        log = pd.read_csv(
            V61_DIGIT_FILE,
            dtype={
                **{f"{p}七码": str for p in ("A", "B", "C")},
                **{f"{p}实际号码": str for p in ("A", "B", "C")},
                **{f"{p}是否命中": str for p in ("A", "B", "C")},
            },
        )
    except (FileNotFoundError, pd.errors.EmptyDataError):
        log = pd.DataFrame(columns=V61_DIGIT_COLS)
    for col in V61_DIGIT_COLS:
        if col not in log.columns:
            log[col] = ""
    # Normalize historic mixed actual-number fields to ordinary Python strings.
    # Pandas Arrow string columns reject integer assignment even when CSV dtype=str.
    for position in ("A", "B", "C"):
        actual_col = f"{position}实际号码"
        log[actual_col] = log[actual_col].astype("object")
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
                log.at[idx, f"{position}实际号码"] = str(actual)
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
        log = pd.read_csv(
            V64_SIX_FILE,
            dtype={
                **{f"{p}六码": str for p in ("A", "B", "C")},
                **{f"{p}实际号码": str for p in ("A", "B", "C")},
                **{f"{p}是否命中": str for p in ("A", "B", "C")},
            },
        )
    except (FileNotFoundError, pd.errors.EmptyDataError):
        log = pd.DataFrame(columns=V64_SIX_COLS)
    for col in V64_SIX_COLS:
        if col not in log.columns:
            log[col] = ""
    # Normalize historic mixed actual-number fields to ordinary Python strings.
    # Pandas Arrow string columns reject integer assignment even when CSV dtype=str.
    for position in ("A", "B", "C"):
        actual_col = f"{position}实际号码"
        log[actual_col] = log[actual_col].astype("object")
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
                log.at[idx, f"{pos}实际号码"] = str(actual)
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


def _six_select_model(values, half_life, points=240):
    """Two-stage strictly causal selection with an untouched confirmation segment.

    First half chooses a candidate; second half only confirms the already chosen
    candidate. If it does not beat the equal blend on BOTH segments, keep blend.
    No future draw is used to construct its own prediction.
    """
    values = np.asarray(values, dtype=int)
    first = max(150, len(values) - points * 2)
    score = {name: [0, 0] for name in SIX_MODEL_NAMES}
    segments = [{name: [0, 0] for name in SIX_MODEL_NAMES} for _ in range(2)]
    indices = list(range(first, len(values), 2))
    midpoint = len(indices) // 2
    for j, i in enumerate(indices):
        candidates = _six_candidates(values[:i], half_life)
        actual = int(values[i])
        bucket = segments[0 if j < midpoint else 1]
        for name in SIX_MODEL_NAMES:
            hit = int(actual in _six_rank(candidates[name]))
            score[name][0] += hit
            score[name][1] += 1
            bucket[name][0] += hit
            bucket[name][1] += 1
    default = "五模型等权"
    if midpoint < 60 or len(indices) - midpoint < 60:
        return default, score, "独立验证样本不足，采用等权模型"
    train = segments[0]
    confirm = segments[1]
    train_rate = lambda name: train[name][0] / train[name][1]
    confirm_rate = lambda name: confirm[name][0] / confirm[name][1]
    candidate = max(SIX_MODEL_NAMES, key=lambda name: (train_rate(name), -SIX_MODEL_NAMES.index(name)))
    # The confirmation segment was not used to pick candidate. Require a
    # meaningful edge in both halves to guard against one-window overfitting.
    if candidate != default and (train_rate(candidate) >= train_rate(default) + .025
            and confirm_rate(candidate) >= confirm_rate(default) + .025
            and confirm_rate(candidate) >= .625):
        return candidate, score, ("前段选模、后段独立验证均超过等权至少2.5个百分点；"
                                  "未来仍可能回落")
    return default, score, "独立验证未确认稳定优势，采用等权模型"


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
    auto_sync = st.checkbox("开启自动同步（每210秒）", value=True)
    st.caption("网页运行时每5秒检查最新一期；发现新开奖立即核对六码/七码。每210秒补充最近历史，校准按每球50条已核对预测执行。")
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
if "last_history_sync" not in st.session_state:
    st.session_state.last_history_sync = time.time()
if "sync_message" not in st.session_state:
    st.session_state.sync_message = "尚未执行本次会话同步"
if "countdown_payload" not in st.session_state:
    st.session_state.countdown_payload = {}


def perform_initial_sync():
    with st.spinner(f"正在同步最近 {import_count} 期开奖数据，请稍候…"):
        incoming = fetch_history(import_count)
        merged = merge_data(st.session_state.history_df, incoming)
        st.session_state.history_df = merged
        save_local(merged)
        st.session_state.last_history_sync = time.time()
        st.session_state.sync_message = f"历史同步完成：当前保存 {len(merged)} 期"


if st.session_state.history_df.empty:
    try:
        perform_initial_sync()
    except Exception as exc:
        st.error(f"首次同步失败：{exc}")
        st.info("请稍后刷新页面，或检查 pc28.world 接口是否可访问。")




def settle_digit_ledgers_fast(draw_rows):
    """Settle existing locked six/seven-digit forecasts before expensive model recalculation."""
    updated = []
    specs = (
        (V64_SIX_FILE, V64_SIX_COLS, 6, "六码"),
        (V61_DIGIT_FILE, V61_DIGIT_COLS, 7, "七码"),
    )
    for path, columns, width, suffix in specs:
        try:
            ledger = pd.read_csv(
                path,
                dtype={
                    **{f"{pos}{suffix}": str for pos in ("A", "B", "C")},
                    **{f"{pos}实际号码": str for pos in ("A", "B", "C")},
                    **{f"{pos}是否命中": str for pos in ("A", "B", "C")},
                },
            )
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        if "目标期号" not in ledger.columns:
            continue
        for col in columns:
            if col not in ledger.columns:
                ledger[col] = ""
        for position in ("A", "B", "C"):
            actual_col = f"{position}实际号码"
            ledger[actual_col] = ledger[actual_col].astype("object")
        target_issues = pd.to_numeric(ledger["目标期号"], errors="coerce")
        changed = False
        for draw in draw_rows:
            issue = int(draw["期号"])
            matching = ledger.index[target_issues.eq(issue)]
            for idx in matching:
                for pos in ("A", "B", "C"):
                    pick = str(ledger.at[idx, f"{pos}{suffix}"]).strip()
                    if len(pick) != width or not pick.isdigit() or len(set(pick)) != width:
                        continue
                    actual = str(int(draw[pos]))
                    verdict = "正确" if actual in pick else "错误"
                    actual_col, verdict_col = f"{pos}实际号码", f"{pos}是否命中"
                    if str(ledger.at[idx, actual_col]) != actual or str(ledger.at[idx, verdict_col]) != verdict:
                        ledger.at[idx, actual_col] = actual
                        ledger.at[idx, verdict_col] = verdict
                        changed = True
        if changed:
            ledger = ledger[columns].sort_values("目标期号").drop_duplicates("目标期号", keep="first")
            ledger.to_csv(path, index=False, encoding="utf-8-sig")
            persist_file(path)
            updated.append(path.name)
    return updated


# 轻量轮询只读取 /draws/latest；不再在每次新开奖时先下载200期历史。
# 新一期先立即入库并触发台账核对；完整补缺仍由独立的210秒任务完成。
@st.fragment(run_every=5 if auto_sync else None)
def sync_clock():
    if auto_sync and not st.session_state.history_df.empty:
        try:
            latest_payload = fetch_latest_payload()
            st.session_state.countdown_payload = latest_payload.get("countdown", {}) or {}
            latest_rows = normalize_rows(unwrap_draws(latest_payload))
            latest = latest_rows[0] if latest_rows else None
            current_max = int(st.session_state.history_df["期号"].max())

            if latest and int(latest["期号"]) > current_max:
                gap = int(latest["期号"]) - current_max
                # 正常只差一期时直接使用 latest 接口结果，避免等待200期历史请求。
                # 若发现跨期跳号，则立即补抓最近200期，避免遗漏已开奖期数。
                if gap == 1:
                    incoming = pd.DataFrame([latest])
                else:
                    recent = normalize_rows(
                        unwrap_draws(api_json("/draws", {"limit": 200}, timeout=10))
                    )
                    incoming = pd.DataFrame(recent) if recent else pd.DataFrame([latest])

                # Persist the new draw first, then settle already-locked ledgers
                # immediately. This runs before the full page recomputes the models.
                st.session_state.history_df = merge_data(
                    st.session_state.history_df, incoming
                )
                save_local(st.session_state.history_df)
                new_draw_rows = (
                    incoming.sort_values("期号").to_dict("records")
                    if not incoming.empty else [latest]
                )
                settled_files = settle_digit_ledgers_fast(new_draw_rows)
                settled_label = (
                    "；六码/七码台账已即时写入数据库"
                    if settled_files else "；本期暂无匹配的待核对台账或无需更新"
                )
                st.session_state.sync_message = (
                    f"发现新期开奖，已同步至第 {int(latest['期号'])} 期{settled_label}"
                )
                st.rerun(scope="app")

            elapsed = time.time() - st.session_state.last_history_sync
            if elapsed >= AUTO_SYNC_SECONDS:
                recent = normalize_rows(
                    unwrap_draws(api_json("/draws", {"limit": 200}, timeout=12))
                )
                if recent:
                    before_max = int(st.session_state.history_df["期号"].max())
                    st.session_state.history_df = merge_data(
                        st.session_state.history_df, pd.DataFrame(recent)
                    )
                    save_local(st.session_state.history_df)
                    after_max = int(st.session_state.history_df["期号"].max())
                else:
                    before_max = after_max = current_max
                st.session_state.last_history_sync = time.time()
                st.session_state.sync_message = (
                    f"210秒历史补同步完成；当前最新第 {after_max} 期"
                )
                if after_max > before_max:
                    st.rerun(scope="app")
        except Exception as exc:
            st.session_state.sync_message = f"自动同步暂时失败：{type(exc).__name__}: {exc}"

    st.caption(
        f"最新开奖检查：每5秒｜历史补同步：每210秒｜{st.session_state.sync_message}"
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

# 大小/单双预测模块已从V6.6.0主流程移除；六码与七码独立运行。
# 旧的大小单双CSV和PostgreSQL记录保留备份，不再生成新记录。

# V6.1 prospective 7-of-10 forecasts; kept as an independent historical ledger.
# Read the existing ledger first so that calibration counts only actual prospective predictions.
# The update function also settles past entries and locks the current issue exactly once.
# First settle existing records without altering a previously locked forecast.
seven_forecast = seven_digit_forecast(df, half_life)
seven_ledger = update_seven_digit_ledger(df, seven_forecast)
digit_calibration = calibrate_digit_models(df, seven_ledger, half_life)
# Calibrated parameters apply to the NEXT issue; existing locked forecasts remain unchanged.
seven_forecast = seven_digit_forecast(df, half_life, digit_calibration)
report_status = build_digit_reports(seven_ledger)

# V6.4 six-digit models are independent of the seven-digit records.
# Start with current calibrated half-life, then lock the six-digit issue once.
six_default_cfg = {pos: {"half_life": half_life} for pos in ("A", "B", "C")}
six_initial = six_digit_forecast(df, six_default_cfg)
six_ledger = update_six_digit_ledger(df, six_initial)
six_calibration = calibrate_six_models(df, six_ledger, half_life)
six_forecast = six_digit_forecast(df, six_calibration)
six_report_status = build_six_reports(six_ledger)

st.subheader("V6.6.1 区域一：三球六码独立预测")
st.caption(f"目标：第 {latest_issue + 1} 期｜六码前段选模+后段独立验证；每球6个不重复数字，公平独立开奖理论覆盖率60%。")
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
st.subheader("V6.6.1 区域二：三球七码独立预测与报告")
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

# 最近开奖结果仅显示开奖号码与和值，不再展示大小/单双预测。
latest_row = df.sort_values("期号").iloc[-1]
a_num, b_num, c_num = int(latest_row["A"]), int(latest_row["B"]), int(latest_row["C"])
draw_sum = int(latest_row["和值"])
st.subheader("最近一期开奖结果")
r1, r2 = st.columns(2)
r1.metric("开奖号码", f"{a_num} + {b_num} + {c_num}")
r2.metric("和值", f"{a_num} + {b_num} + {c_num} = {draw_sum}")
st.caption(f"开奖期号：第 {latest_issue} 期｜北京时间：{format_beijing_time(latest_time)}")

st.subheader("下一期开奖倒计时")

@st.fragment(run_every=5)
def draw_countdown():
    try:
        countdown_data = st.session_state.get("countdown_payload", {})
        if not isinstance(countdown_data, dict):
            countdown_data = {}
        seconds_value = countdown_data.get("seconds")
        next_draw_at = countdown_data.get("next_draw_at")
        fetched_at = beijing_now()

        if next_draw_at is not None:
            target_timestamp = float(next_draw_at)
            if target_timestamp > 10_000_000_000:
                target_timestamp /= 1000.0
            target_dt = datetime.fromtimestamp(target_timestamp, tz=BEIJING_TZ)
            seconds_left = max(0, int((target_dt - fetched_at).total_seconds()))
            time_text = target_dt.strftime("%Y-%m-%d %H:%M:%S")
            source_label = "开奖时间来自 pc28.world 实时接口"
        elif seconds_value is not None:
            seconds_left = max(0, int(float(seconds_value)))
            target_dt = fetched_at + timedelta(seconds=seconds_left)
            time_text = target_dt.strftime("%Y-%m-%d %H:%M:%S")
            source_label = "倒计时秒数来自 pc28.world 实时接口"
        else:
            st.info("等待下一次5秒轮询取得数据源倒计时。")
            return

        mins, secs = divmod(seconds_left, 60)
        hours, mins = divmod(mins, 60)
        countdown_text = f"{hours:02d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"
        st.metric("实时预计开奖倒计时", countdown_text)
        st.caption(f"预计开奖时间（北京时间）：{time_text}｜{source_label}")
    except Exception as exc:
        st.warning(f"暂时无法显示实时倒计时：{type(exc).__name__}: {exc}")


draw_countdown()

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
    "本应用仅作三球六码/七码数据统计与研究，不提供投注功能；每球每累计50条已核对事前预测独立校准，不保证未来覆盖率提高。"
)
st.caption(
    "重要：Streamlit Cloud 本地文件可能在重启或重新部署后丢失；"
    "请定期下载 CSV 备份。如已配置 PostgreSQL 则使用云数据库；否则本地存储不保证持久。"
)


# V6.5.3: strictly prospective streak statistics, no fabricated past forecasts.
def _streak_analysis(log, position):
    col = f"{position}是否命中"
    if log is None or log.empty or col not in log.columns:
        return None
    work = log.copy()
    work["目标期号"] = pd.to_numeric(work["目标期号"], errors="coerce")
    work = work.dropna(subset=["目标期号"]).sort_values("目标期号")
    work = work.drop_duplicates("目标期号", keep="first")
    settled = work[work[col].isin(("正确", "错误"))].copy()
    if settled.empty:
        return None
    outcomes = list(zip(settled["目标期号"].astype(int), settled[col].tolist()))
    runs = []
    prev_issue, previous_result, length, start = None, None, 0, None
    for issue, result in outcomes:
        if result == previous_result and prev_issue is not None and issue == prev_issue + 1:
            length += 1
        else:
            if length:
                runs.append({"结果": previous_result, "连续期数": length,
                             "起始期号": start, "结束期号": prev_issue})
            length, start = 1, issue
        previous_result, prev_issue = result, issue
    if length:
        runs.append({"结果": previous_result, "连续期数": length,
                     "起始期号": start, "结束期号": prev_issue})
    # A pending forecast after the last settled issue does not invalidate the settled streak.
    last_run = runs[-1]
    wins = [r["连续期数"] for r in runs if r["结果"] == "正确"]
    losses = [r["连续期数"] for r in runs if r["结果"] == "错误"]
    n = len(settled)
    hits = int((settled[col] == "正确").sum())
    return {
        "n": n, "hits": hits, "rate": hits / n,
        "current_win": last_run["连续期数"] if last_run["结果"] == "正确" else 0,
        "current_loss": last_run["连续期数"] if last_run["结果"] == "错误" else 0,
        "max_win": max(wins, default=0), "max_loss": max(losses, default=0),
        "runs": pd.DataFrame(runs), "settled": settled,
    }

st.header("📈 连续命中统计与模型状态观察")
st.caption("只使用开奖前保存、开奖后已核对的预测记录。连续期数必须期号相邻；漏期会中断连续统计。")
kind = st.radio("统计类型", ["七码", "六码"], horizontal=True, key="streak_kind")
source_log = seven_ledger if kind == "七码" else six_ledger
summary_rows = []
for position in ("A", "B", "C"):
    stat = _streak_analysis(source_log, position)
    if stat is None:
        summary_rows.append({"球位": position, "已核对期数": 0, "命中率": "—", "当前连中": 0,
                             "当前连错": 0, "最长连中": 0, "最长连错": 0})
        continue
    summary_rows.append({"球位": position, "已核对期数": stat["n"],
                         "命中率": f"{stat['rate']:.1%}", "当前连中": stat["current_win"],
                         "当前连错": stat["current_loss"], "最长连中": stat["max_win"],
                         "最长连错": stat["max_loss"]})
st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

# V6.5.3: combined A/B/C rankings and complete winning-run history.
# These are derived from the persisted prospective ledgers, not backfilled forecasts.
st.subheader(f"🏆 三球{kind}当前连中排行榜")
rank_rows = []
all_win_rows = []
for ball in ("A", "B", "C"):
    ball_stat = _streak_analysis(source_log, ball)
    if ball_stat is None:
        rank_rows.append({"球位": ball, "当前连中": 0, "历史最长连中": 0,
                          "已核对期数": 0, "最近50条命中率": "—"})
        continue
    recent = ball_stat["settled"].tail(50)
    recent_rate = (recent[f"{ball}是否命中"] == "正确").mean()
    rank_rows.append({"球位": ball, "当前连中": ball_stat["current_win"],
                      "历史最长连中": ball_stat["max_win"],
                      "已核对期数": ball_stat["n"],
                      "最近50条命中率": f"{recent_rate:.1%}"})
    ball_runs = ball_stat["runs"]
    for idx, record in ball_runs.iterrows():
        if record["结果"] != "正确":
            continue
        all_win_rows.append({"球位": ball, "起始期号": int(record["起始期号"]),
                             "结束期号": int(record["结束期号"]),
                             "连续命中期数": int(record["连续期数"]),
                             "状态": "当前连中" if idx == ball_runs.index[-1] else "已结束"})
rank_df = pd.DataFrame(rank_rows)
current_rank = rank_df.sort_values(["当前连中", "历史最长连中", "球位"],
                                   ascending=[False, False, True]).reset_index(drop=True)
current_rank.insert(0, "排名", range(1, len(current_rank) + 1))
st.dataframe(current_rank, hide_index=True, use_container_width=True)
st.subheader(f"🥇 三球{kind}历史最高连中排行榜")
history_rank = rank_df.sort_values(["历史最长连中", "当前连中", "球位"],
                                   ascending=[False, False, True]).reset_index(drop=True)
history_rank.insert(0, "排名", range(1, len(history_rank) + 1))
st.dataframe(history_rank, hide_index=True, use_container_width=True)

st.subheader(f"📚 A/B/C {kind}全部连续命中纪录")
if all_win_rows:
    win_history = pd.DataFrame(all_win_rows)
    # Keep even one-hit runs so no consecutive-correct segment is silently omitted.
    win_history = win_history.sort_values(["连续命中期数", "结束期号", "球位"],
                                          ascending=[False, False, True]).reset_index(drop=True)
    st.caption("按连续命中期数从高到低排列；同次数按结束期号排序。含1期记录；漏期自动中断。")
    st.dataframe(win_history, hide_index=True, use_container_width=True)
    st.download_button(f"下载三球{kind}全部连中纪录 CSV",
                       win_history.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"canada28_{'seven' if kind == '七码' else 'six'}_all_ball_win_streaks.csv",
                       mime="text/csv", key=f"all_ball_streak_download_{kind}")
else:
    st.info("目前没有已核对的连续命中纪录。")
st.caption("排行榜只反映已发生的真实事前预测命中情况，不表示未来更容易连续命中。")

position = st.selectbox("查看球位详情", ["A", "B", "C"], key="streak_position")
stat = _streak_analysis(source_log, position)
if stat is None:
    st.info("该球位暂无已核对的事前预测。")
else:
    st.subheader(f"{position}球 · {kind} 连中记录")
    wins = stat["runs"][stat["runs"]["结果"] == "正确"].copy()
    distribution = pd.DataFrame([
        {"至少连续命中": f"{threshold}期", "出现次数": int((wins["连续期数"] >= threshold).sum())}
        for threshold in (2, 3, 5, 7, 10)
    ])
    st.dataframe(distribution, hide_index=True, use_container_width=True)
    st.dataframe(stat["runs"].iloc[::-1].head(50), hide_index=True, use_container_width=True)
    st.subheader("最近窗口的已验证命中率")
    settled = stat["settled"]
    col = f"{position}是否命中"
    window_rows = []
    for window in (50, 100, 200):
        recent = settled.tail(window)
        n = len(recent)
        window_rows.append({"窗口": f"最近{window}条已核对预测", "实际样本": n,
                            "命中率": f"{(recent[col] == '正确').mean():.1%}" if n else "—",
                            "理论覆盖率": "70%" if kind == "七码" else "60%"})
    st.dataframe(pd.DataFrame(window_rows), hide_index=True, use_container_width=True)
    st.caption("窗口统计仅描述已记录预测的表现，不等于对未来的置信概率；没有可验证的方法判断下一期何时‘开始准’。")
    st.download_button("下载连续统计CSV", stat["runs"].to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"canada28_{kind}_{position}_streaks.csv", mime="text/csv")


# V6.5.4: longest-run drilldown and comparable 50/100/200-forecast trend tables.
# All calculations use locked prospective records, never reconstructed forecasts.
st.header("🔎 V6.5.4 · 三球连中明细与分窗口走势")
st.caption("每个窗口以该球位最近N条已核对的事前预测为样本；期号不相邻时连中中断。窗口内最长连中可能短于完整历史最长连中。")


def _v654_window_summary(stat, ball, window):
    if stat is None:
        return {"球位": ball, "窗口": window, "有效期数": 0, "命中": 0,
                "命中率": "—", "窗口最长连中": 0, "窗口末尾连中": 0,
                "窗口最长连错": 0}
    sample = stat["settled"].tail(window).copy()
    if sample.empty:
        return {"球位": ball, "窗口": window, "有效期数": 0, "命中": 0,
                "命中率": "—", "窗口最长连中": 0, "窗口末尾连中": 0,
                "窗口最长连错": 0}
    # _streak_analysis recomputes runs on this window only, preserving issue gaps.
    window_stat = _streak_analysis(sample, ball)
    return {"球位": ball, "窗口": window, "有效期数": window_stat["n"],
            "命中": window_stat["hits"], "命中率": f"{window_stat['rate']:.1%}",
            "窗口最长连中": window_stat["max_win"],
            "窗口末尾连中": window_stat["current_win"],
            "窗口最长连错": window_stat["max_loss"]}


v654_stats = {ball: _streak_analysis(source_log, ball) for ball in ("A", "B", "C")}
v654_rows = [_v654_window_summary(v654_stats[ball], ball, window)
             for window in (50, 100, 200) for ball in ("A", "B", "C")]
v654_table = pd.DataFrame(v654_rows)
st.subheader(f"📊 A/B/C {kind}最近50、100、200期走势对比")
for window in (50, 100, 200):
    st.markdown(f"**最近{window}条已核对预测**")
    st.dataframe(v654_table[v654_table["窗口"] == window].drop(columns=["窗口"]),
                 hide_index=True, use_container_width=True)
st.download_button("下载三球分窗口走势 CSV",
                   v654_table.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"canada28_v654_{'seven' if kind == '七码' else 'six'}_window_trends.csv",
                   mime="text/csv", key=f"v654_window_csv_{kind}")

st.subheader(f"🏅 A/B/C {kind}历史最长连中：起止期号与逐期明细")
longest_rows = []
for ball, ball_stat in v654_stats.items():
    if ball_stat is None:
        continue
    runs = ball_stat["runs"]
    winners = runs[(runs["结果"] == "正确") & (runs["连续期数"] == ball_stat["max_win"])]
    for _, run in winners.iterrows():
        longest_rows.append({"球位": ball, "最长连中期数": int(run["连续期数"]),
                             "起始期号": int(run["起始期号"]),
                             "结束期号": int(run["结束期号"])})
if longest_rows:
    longest_df = pd.DataFrame(longest_rows).sort_values(
        ["最长连中期数", "结束期号", "球位"], ascending=[False, False, True])
    st.caption("并列最长纪录全部保留；每段连中仅统计真实事前保存且已核对的相邻期号。")
    st.dataframe(longest_df, hide_index=True, use_container_width=True)
    st.download_button("下载历史最长连中纪录 CSV",
                       longest_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"canada28_v654_{'seven' if kind == '七码' else 'six'}_longest_runs.csv",
                       mime="text/csv", key=f"v654_longest_csv_{kind}")
    detail_ball = st.selectbox("查看最长连中逐期明细的球位", ["A", "B", "C"], key="v654_detail_ball")
    ball_runs = longest_df[longest_df["球位"] == detail_ball]
    if ball_runs.empty:
        st.info("该球位暂无已核对的连中纪录。")
    else:
        choices = [f"{int(r['起始期号'])} — {int(r['结束期号'])}（{int(r['最长连中期数'])}期）"
                   for _, r in ball_runs.iterrows()]
        selected = st.selectbox("选择连中区间", range(len(choices)),
                                format_func=lambda i: choices[i], key="v654_detail_range")
        chosen = ball_runs.iloc[selected]
        ball_log = v654_stats[detail_ball]["settled"].copy()
        issue = pd.to_numeric(ball_log["目标期号"], errors="coerce")
        detail = ball_log[issue.between(int(chosen["起始期号"]), int(chosen["结束期号"]))].copy()
        columns = [c for c in ("目标期号", "预测时间", f"{detail_ball}{kind}",
                               f"{detail_ball}实际号码", f"{detail_ball}是否命中") if c in detail.columns]
        st.dataframe(detail[columns].sort_values("目标期号"), hide_index=True, use_container_width=True)
        st.download_button("下载本段逐期明细 CSV",
                           detail[columns].sort_values("目标期号").to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"canada28_v654_{kind}_{detail_ball}_longest_detail.csv",
                           mime="text/csv", key=f"v654_detail_csv_{kind}_{detail_ball}")
else:
    st.info("暂无已核对的连续命中纪录。")
st.caption("历史连中、最近窗口命中率都不能可靠预测下一期是否命中；独立随机开奖不会因为之前连中或连错而改变概率。")
