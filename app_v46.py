# -*- coding: utf-8 -*-
"""Canada28 Matrix AI V4.6
历史统计与大小/单双概率研究工具。非投注软件，不保证预测准确。
"""
import os
import time
from datetime import datetime
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

st.set_page_config(
    page_title="Canada28 Matrix AI V4.6",
    page_icon="📊",
    layout="wide",
)

st.title("Canada28 Matrix AI V4.6")
st.caption(
    "历史数据驱动的大小 / 单双概率研究。数据来自 pc28.world；"
    "统计概率不代表能够预测随机开奖结果，也不构成投注建议。"
)


def api_json(path, params=None, timeout=15):
    response = requests.get(
        f"{API_BASE}{path}",
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Canada28-Matrix-AI/4.6"},
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


def fetch_latest():
    data = api_json("/draws/latest")
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


def recommendation(p, label_yes, label_no, min_edge=0.03):
    edge = abs(p - 0.5)
    if edge < min_edge:
        return "暂不推荐", edge
    return (label_yes if p > 0.5 else label_no), edge


def rolling_backtest(df, half_life=500, prior_strength=200, tests=300):
    """逐期向前测试；每期只使用它之前的数据。"""
    if len(df) < 120:
        return None
    start = max(100, len(df) - int(tests))
    big_hits, odd_hits, n = 0, 0, 0
    for i in range(start, len(df)):
        hist = df.iloc[:i]
        probs = calculate_probabilities(hist, half_life, prior_strength)
        actual_sum = int(df.iloc[i]["和值"])
        pred_big = probs["big"]["final"] >= 0.5
        pred_odd = probs["odd"]["final"] >= 0.5
        big_hits += int(pred_big == (actual_sum >= 14))
        odd_hits += int(pred_odd == (actual_sum % 2 == 1))
        n += 1
    return {
        "n": n,
        "big": big_hits / n if n else 0,
        "odd": odd_hits / n if n else 0,
    }


# Sidebar controls
with st.sidebar:
    st.header("系统设置")
    import_count = st.selectbox("首次同步历史期数", [1000, 3000, 5000], index=2)
    half_life = st.slider("近期权重半衰期（期）", 100, 2000, 500, 50)
    prior_strength = st.slider("先验平滑强度", 0, 1000, 200, 25)
    min_edge = st.slider("推荐最低概率优势", 1, 10, 3) / 100
    auto_sync = st.checkbox("开启自动同步（每210秒）", value=True)
    st.caption("自动同步只在网页打开且应用运行时执行。")
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
# Periodic fragment refresh. The sync check must live INSIDE the fragment,
# otherwise only the caption would refresh and no network request would run.
@st.fragment(run_every=AUTO_SYNC_SECONDS if auto_sync else None)
def sync_clock():
    if auto_sync and not st.session_state.history_df.empty:
        elapsed = time.time() - st.session_state.last_sync
        if elapsed >= AUTO_SYNC_SECONDS:
            try:
                latest = fetch_latest()
                current_max = int(st.session_state.history_df["期号"].max())
                if latest and int(latest["期号"]) > current_max:
                    # 拉取最近一页并合并；期号去重，保留完整已存历史。
                    recent = normalize_rows(
                        unwrap_draws(api_json("/draws", {"limit": 200}))
                    )
                    new_df = pd.DataFrame(recent)
                    st.session_state.history_df = merge_data(
                        st.session_state.history_df, new_df
                    )
                    save_local(st.session_state.history_df)
                    st.session_state.sync_message = (
                        f"发现新期开奖，已更新至第 {int(latest['期号'])} 期"
                    )
                    st.session_state.last_sync = time.time()
                    st.rerun()
                else:
                    st.session_state.sync_message = "已检查最新开奖，暂无新期号"
                    st.session_state.last_sync = time.time()
            except Exception as exc:
                st.session_state.sync_message = f"自动同步暂时失败：{exc}"
                st.session_state.last_sync = time.time()
    st.caption(
        f"自动同步间隔：210秒（3分30秒）｜{st.session_state.sync_message}"
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
probs = calculate_probabilities(df, half_life, prior_strength)
p_big = probs["big"]["final"]
p_odd = probs["odd"]["final"]
big_pick, big_edge = recommendation(p_big, "大", "小", min_edge)
odd_pick, odd_edge = recommendation(p_odd, "单", "双", min_edge)

st.subheader("下一期大小 / 单双分析")
st.caption(
    f"当前数据截至：第 {latest_issue} 期"
    + (f"｜开奖时间：{latest_time}" if latest_time else "")
    + f"｜有效历史：{len(df):,} 期"
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 大小推荐")
    st.metric("大（14–27）", f"{p_big * 100:.1f}%")
    st.progress(float(np.clip(p_big, 0, 1)))
    st.metric("小（0–13）", f"{(1-p_big) * 100:.1f}%")
    st.markdown(f"**分析结果：{big_pick}**")
    st.caption(f"与50%基准的概率差：{big_edge*100:.1f}个百分点")
with c2:
    st.markdown("#### 单双推荐")
    st.metric("单", f"{p_odd * 100:.1f}%")
    st.progress(float(np.clip(p_odd, 0, 1)))
    st.metric("双", f"{(1-p_odd) * 100:.1f}%")
    st.markdown(f"**分析结果：{odd_pick}**")
    st.caption(f"与50%基准的概率差：{odd_edge*100:.1f}个百分点")

st.info(
    "推荐规则：模型概率与50%基准至少相差设定门槛才显示方向；"
    "否则显示“暂不推荐”。这只是统计筛选规则，不代表该方向有可验证的盈利优势。"
)

with st.expander("查看模型分项概率", expanded=False):
    detail_rows = []
    for target, title, labels in [
        ("big", "大小", ("大", "小")),
        ("odd", "单双", ("单", "双")),
    ]:
        result = probs[target]
        for key, name in [
            ("all", "全历史平滑"),
            ("500", "最近500期平滑"),
            ("100", "最近100期平滑"),
            ("weighted", "近期加权"),
            ("final", "综合结果"),
        ]:
            p = result[key]
            detail_rows.append({
                "玩法": title,
                "统计模型": name,
                labels[0]: f"{p*100:.1f}%",
                labels[1]: f"{(1-p)*100:.1f}%",
            })
    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

with st.expander("历史回测（逐期向前验证）", expanded=False):
    if st.button("运行最近300期回测"):
        with st.spinner("正在逐期回测，请稍候…"):
            result = rolling_backtest(
                df, half_life, prior_strength, tests=300
            )
        if result:
            st.write(f"测试期数：{result['n']}")
            st.metric("大小方向命中率", f"{result['big']*100:.1f}%")
            st.metric("单双方向命中率", f"{result['odd']*100:.1f}%")
            st.caption(
                "每一期只使用此前历史数据计算，避免把该期结果提前用于预测。"
                "回测命中率不保证未来表现；若要严谨评估，还需长期留出样本和基准比较。"
            )
        else:
            st.warning("有效历史数据不足，至少需要120期。")

with st.expander("最近开奖记录", expanded=False):
    st.dataframe(df.tail(30).iloc[::-1], use_container_width=True, hide_index=True)

csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "下载历史数据 CSV 备份",
    data=csv_bytes,
    file_name="canada28_history_backup.csv",
    mime="text/csv",
)

st.caption(
    "数据来自 pc28.world，使用时请注明并链接 https://pc28.world。"
    "本应用仅作数据统计与研究，不提供投注功能。"
)
st.caption(
    "重要：Streamlit Cloud 本地文件可能在重启或重新部署后丢失；"
    "请定期下载 CSV 备份。此版本尚未接入永久云数据库。"
)
