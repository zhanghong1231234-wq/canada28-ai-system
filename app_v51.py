# -*- coding: utf-8 -*-
"""Canada28 Matrix AI V5.1
历史统计与大小/单双概率研究工具。非投注软件，不保证预测准确。
"""
import os
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
    page_title="Canada28 Matrix AI V5.1",
    page_icon="📊",
    layout="wide",
)

st.title("Canada28 Matrix AI V5.1")
st.caption(
    "历史数据驱动的大小 / 单双概率研究。数据来自 pc28.world；"
    "统计概率不代表能够预测随机开奖结果，也不构成投注建议。"
)


def api_json(path, params=None, timeout=15):
    response = requests.get(
        f"{API_BASE}{path}",
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Canada28-Matrix-AI/5.0"},
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
        X, y = X[-1500:], y[-1500:]
        try:
            from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
            forest = RandomForestClassifier(n_estimators=60, max_depth=4,
                                            min_samples_leaf=30, random_state=2026, n_jobs=1)
            boost = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7,
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


def rolling_backtest(df, half_life=500, prior_strength=200, tests=300):
    """逐期向前测试；每期只使用它之前的数据，并分别评估各统计模型。"""
    if len(df) < 120:
        return None
    start = max(100, len(df) - int(tests))
    keys = ["all", "500", "100", "weighted", "final"]
    names = {"all": "全历史平滑", "500": "最近500期平滑",
             "100": "最近100期平滑", "weighted": "近期加权", "final": "综合结果"}
    stats = {k: {"bh": 0, "oh": 0, "bb": 0.0, "ob": 0.0} for k in keys}
    n = 0
    for i in range(start, len(df)):
        p = calculate_probabilities(df.iloc[:i], half_life, prior_strength)
        total = int(df.iloc[i]["和值"])
        actual_big, actual_odd = int(total >= 14), int(total % 2 == 1)
        for key in keys:
            pb, po = float(p["big"][key]), float(p["odd"][key])
            stats[key]["bh"] += int((pb >= 0.5) == bool(actual_big))
            stats[key]["oh"] += int((po >= 0.5) == bool(actual_odd))
            stats[key]["bb"] += (pb - actual_big) ** 2
            stats[key]["ob"] += (po - actual_odd) ** 2
        n += 1
    rows = []
    for key in keys:
        s = stats[key]
        rows.append({
            "模型": names[key],
            "大小命中率": s["bh"] / n,
            "单双命中率": s["oh"] / n,
            "大小Brier分数": s["bb"] / n,
            "单双Brier分数": s["ob"] / n,
        })
    return {"n": n, "models": pd.DataFrame(rows)}


FORECAST_FILE = DATA_DIR / "canada28_forecast_log.csv"
CALIBRATION_STATE_FILE = DATA_DIR / "calibration_state.json"
CALIBRATION_LOG_FILE = DATA_DIR / "calibration_history.csv"
CALIBRATION_BATCH_SIZE = 50


def update_forecast_ledger(df, probabilities):
    """每个目标期只保存首次预测；开奖后分别核对单双与大小。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "预测时间", "目标期号", "基于最新期号",
        "预测单双", "单双概率单", "实际单双", "单双结果",
        "预测大小", "大小概率大", "实际大小", "大小结果"
    ]
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
        except (TypeError, ValueError, KeyError):
            continue

    log = log[cols].sort_values("目标期号")
    log.to_csv(FORECAST_FILE, index=False, encoding="utf-8-sig")
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

    state.update({
        "last_calibrated_issue": last_issue,
        "half_life": active_half,
        "prior_strength": active_prior,
        "completed_batches": int(state.get("completed_batches", 0)) + batches,
    })
    CALIBRATION_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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
    st.caption("网页运行时每5秒检查最新开奖，每210秒补充历史。每50期校准旧版平滑参数；九模型权重暂固定。")
    st.caption("数据来源：pc28.world（第三方只读数据接口）")

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
ledger_df = update_forecast_ledger(df, probs)
p_big = probs["big"]["final"]
p_odd = probs["odd"]["final"]
big_pick = "大" if p_big >= 0.5 else "小"
big_edge = abs(p_big - 0.5)
odd_pick = "单" if p_odd >= 0.5 else "双"
odd_edge = abs(p_odd - 0.5)

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
        for key, name in [*MODEL_NAMES.items(), ("final", "九模型综合")]:
            p = result[key]
            detail_rows.append({
                "玩法": title,
                "统计模型": name,
                labels[0]: f"{p*100:.1f}%",
                labels[1]: f"{(1-p)*100:.1f}%",
            })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

with st.expander("历史回测（逐期向前验证）", expanded=False):
    test_count = st.selectbox("回测最近多少期", [100, 300, 500, 1000], index=1)
    if st.button("运行独立模型回测"):
        with st.spinner("正在运行时间顺序基准回测，请稍候…"):
            result = rolling_backtest(df, half_life, prior_strength, tests=test_count)
        if result:
            st.write(f"测试期数：{result['n']}")
            display = result["models"].copy()
            for col in ["大小命中率", "单双命中率"]:
                display[col] = display[col].map(lambda x: f"{x*100:.1f}%")
            for col in ["大小Brier分数", "单双Brier分数"]:
                display[col] = display[col].map(lambda x: f"{x:.4f}")
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption("此处是旧版五种统计基准的逐期回测，并非九模型综合的独立测试。Brier 0.2500 为公平随机基准；不能把此表当作 V5.0 综合模型胜率。")
        else:
            st.warning("有效历史数据不足，至少需要120期。")

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
    "本应用仅作数据统计与研究，不提供投注功能；每50期仅校准统计平滑参数，九模型权重暂固定，不保证未来命中率提高。"
)
st.caption(
    "重要：Streamlit Cloud 本地文件可能在重启或重新部署后丢失；"
    "请定期下载 CSV 备份。此版本尚未接入永久云数据库。"
)
