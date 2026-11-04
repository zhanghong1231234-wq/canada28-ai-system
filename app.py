# -*- coding: utf-8 -*-
"""Canada28 Matrix AI V4.3 — historical analysis and walk-forward model comparison."""
import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Canada28 Matrix AI V4.3", page_icon="📊", layout="wide")
st.title("Canada28 Matrix AI V4.3")
st.caption("真实历史数据分析、三模型时间顺序回测与概率对比。历史统计不保证预测未来开奖结果。")

COL_ALIASES = {
    "期号": ["期号", "期数", "开奖期号", "draw_id", "issue", "period"],
    "A": ["A", "号码1", "数字1", "开奖号1", "num1"],
    "B": ["B", "号码2", "数字2", "开奖号2", "num2"],
    "C": ["C", "号码3", "数字3", "开奖号3", "num3"],
}
SUMS = np.arange(28)
THEORY_COUNTS = np.bincount(
    [a + b + c for a in range(10) for b in range(10) for c in range(10)],
    minlength=28,
)
THEORY = THEORY_COUNTS / 1000.0

def read_upload(upload):
    raw = upload.getvalue()
    name = upload.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    try:
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(raw), encoding="gbk")

def standardize_columns(frame):
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    rename = {}
    for target, aliases in COL_ALIASES.items():
        found = next((lookup[a.lower()] for a in aliases if a.lower() in lookup), None)
        if found is not None:
            rename[found] = target
    out = frame.rename(columns=rename).copy()
    missing = [c for c in ["A", "B", "C"] if c not in out.columns]
    if missing:
        raise ValueError("找不到 A、B、C 三个号码列。支持列名：A/B/C 或 号码1/号码2/号码3。")
    if "期号" not in out.columns:
        out.insert(0, "期号", np.arange(1, len(out) + 1).astype(str))
    return out[["期号", "A", "B", "C"]]

def clean_data(raw):
    df = standardize_columns(raw)
    df["期号"] = df["期号"].astype(str).str.strip()
    for col in ["A", "B", "C"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    original = len(df)
    df = df.dropna(subset=["期号", "A", "B", "C"]).copy()
    valid = df[["A", "B", "C"]].apply(lambda s: s.between(0, 9) & (s % 1 == 0)).all(axis=1)
    invalid = int((~valid).sum())
    df = df[valid].copy()
    duplicates = int(df.duplicated(subset=["期号"], keep="last").sum())
    df = df.drop_duplicates(subset=["期号"], keep="last")
    for col in ["A", "B", "C"]:
        df[col] = df[col].astype(int)
    df["和值"] = df[["A", "B", "C"]].sum(axis=1)
    df["大小"] = np.where(df["和值"] >= 14, "大", "小")
    df["单双"] = np.where(df["和值"] % 2 == 0, "双", "单")
    return df.reset_index(drop=True), original, invalid, duplicates

def smoothed_recent_probability(history_sums, window=100, strength=100.0):
    recent = np.asarray(history_sums[-window:], dtype=int)
    counts = np.bincount(recent, minlength=28).astype(float)
    return (counts + strength * THEORY) / (len(recent) + strength)

def rolling_dynamic_probability(history_sums, window=30, strength=30.0):
    """滚动窗口模型：近期窗口频率向理论分布收缩。"""
    return smoothed_recent_probability(history_sums, window, strength)

def score_probabilities(probabilities, actual):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0)
    actual = np.asarray(actual, dtype=int)
    chosen = p[np.arange(len(actual)), actual]
    onehot = np.eye(28)[actual]
    return float(-np.mean(np.log(chosen))), float(np.mean(np.sum((p-onehot)**2, axis=1)))

def evaluate_three_models(sums, window=100, strength=100.0, dynamic_window=30, dynamic_strength=30.0):
    """严格逐期前向测试：每次只传入当前期之前的数据。"""
    sums = np.asarray(sums, dtype=int)
    split = max(1, int(len(sums) * 0.8))
    if len(sums) - split < 1:
        return None
    rows, probs = [], {"A 理论基准": [], "B 近期平滑": [], "C 滚动动态": []}
    for idx in range(split, len(sums)):
        hist = sums[:idx]
        p_a = THEORY.copy()
        p_b = smoothed_recent_probability(hist, window, strength)
        p_c = rolling_dynamic_probability(hist, dynamic_window, dynamic_strength)
        actual = int(sums[idx])
        models = [("A 理论基准", p_a), ("B 近期平滑", p_b), ("C 滚动动态", p_c)]
        row = {"期号索引": idx + 1, "预测前已知历史期数": idx, "实际和值": actual}
        for name, p in models:
            pred = int(np.flatnonzero(np.isclose(p, p.max()))[0])
            row[f"{name}预测和值"] = pred
            row[f"{name}是否命中"] = "命中" if pred == actual else "未命中"
            row[f"{name}和值差"] = abs(pred - actual)
            probs[name].append(p)
        rows.append(row)
    detail = pd.DataFrame(rows)
    actual = sums[split:]
    summary = []
    for name in probs:
        pred_col = f"{name}预测和值"
        hit_col = f"{name}是否命中"
        diff_col = f"{name}和值差"
        hits = int((detail[hit_col] == "命中").sum())
        ll, bs = score_probabilities(probs[name], actual)
        summary.append({
            "模型": name, "完全命中": hits, "测试期数": len(actual),
            "命中率(%)": hits / len(actual) * 100,
            "平均和值差": float(detail[diff_col].mean()),
            "Log Loss（越低越好）": ll, "Brier Score（越低越好）": bs,
        })
    return split, detail, pd.DataFrame(summary)

upload = st.sidebar.file_uploader("上传真实历史开奖数据（Excel / CSV）", type=["xlsx", "xls", "csv"])
st.sidebar.markdown("数据至少包含 A、B、C 三列；也支持「号码1、号码2、号码3」。")
window = st.sidebar.select_slider("近期统计窗口", options=[20, 50, 100, 200, 300, 500], value=100)
strength = st.sidebar.slider("近期频率平滑强度", min_value=10, max_value=500, value=100, step=10)
dynamic_window = st.sidebar.select_slider("动态模型滚动窗口", options=[10, 20, 30, 50, 100], value=30)
dynamic_strength = st.sidebar.slider("动态模型平滑强度", min_value=5, max_value=200, value=30, step=5)
mc_runs = st.sidebar.selectbox("蒙特卡洛模拟次数", [100_000, 500_000, 1_000_000], index=2)

if upload is None:
    st.info("请从左侧上传 Excel 或 CSV 历史开奖文件。系统不会把随机模拟数据当作真实历史数据。")
    template = pd.DataFrame(columns=["期号", "A", "B", "C"])
    st.download_button("下载数据模板 CSV", template.to_csv(index=False).encode("utf-8-sig"),
                       "Canada28_history_template.csv", "text/csv")
    st.stop()

try:
    raw = read_upload(upload)
    data, original_rows, invalid_rows, duplicate_rows = clean_data(raw)
except Exception as exc:
    st.error(f"数据读取或格式检查失败：{exc}")
    st.write("当前文件列名：", list(raw.columns) if "raw" in locals() else "无法读取")
    st.stop()

if data.empty:
    st.error("没有可分析的有效记录，请检查 Excel 数据。")
    st.stop()

st.success(f"成功导入 {len(data):,} 期有效记录。请确认 Excel 原始顺序为「旧 → 新」。")
m1, m2, m3, m4 = st.columns(4)
m1.metric("有效期数", f"{len(data):,}")
m2.metric("无效号码记录", f"{invalid_rows:,}")
m3.metric("重复期号记录", f"{duplicate_rows:,}")
m4.metric("平均和值", f"{data['和值'].mean():.2f}")
with st.expander("数据检查与最近记录", expanded=False):
    st.write(f"原始记录：{original_rows:,}；清洗后：{len(data):,}")
    st.dataframe(data.tail(30), use_container_width=True, hide_index=True)

full_counts = data["和值"].value_counts().reindex(SUMS, fill_value=0)
recent = data["和值"].tail(window)
recent_counts = recent.value_counts().reindex(SUMS, fill_value=0)
recent_prob = smoothed_recent_probability(data["和值"].to_numpy(), window, strength)
matrix = pd.DataFrame({
    "和值": SUMS, "理论概率(%)": THEORY * 100,
    "历史频率(%)": full_counts.to_numpy() / len(data) * 100,
    f"近{min(window, len(data))}期频率(%)": recent_counts.to_numpy() / len(recent) * 100,
    "平滑近期参考概率(%)": recent_prob * 100,
})
st.subheader("1. 0–27 和值概率矩阵")
st.dataframe(matrix.style.format({c: "{:.3f}" for c in matrix.columns if c != "和值"}), use_container_width=True, hide_index=True)
st.line_chart(matrix.set_index("和值")[["理论概率(%)", "历史频率(%)", "平滑近期参考概率(%)"]], use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.subheader("2. A / B / C 数字分布")
    digit_matrix = pd.DataFrame({"数字": range(10)})
    for col in ["A", "B", "C"]:
        counts = data[col].value_counts().reindex(range(10), fill_value=0)
        digit_matrix[f"{col}次数"] = counts.to_numpy()
        digit_matrix[f"{col}频率(%)"] = counts.to_numpy() / len(data) * 100
    st.dataframe(digit_matrix, use_container_width=True, hide_index=True)
with c2:
    st.subheader("3. 大小 / 单双统计")
    category = pd.DataFrame({
        "类别": ["大(14–27)", "小(0–13)", "单", "双"],
        "次数": [int((data["和值"] >= 14).sum()), int((data["和值"] <= 13).sum()),
                 int((data["和值"] % 2 == 1).sum()), int((data["和值"] % 2 == 0).sum())],
    })
    category["频率(%)"] = category["次数"] / len(data) * 100
    st.dataframe(category, use_container_width=True, hide_index=True)

st.subheader("4. 近期动态统计（20 / 50 / 100 期）")
trend_rows = []
for n in [20, 50, 100]:
    part = data["和值"].tail(min(n, len(data)))
    counts = part.value_counts()
    trend_rows.append({"统计窗口": f"近{len(part)}期", "样本数": len(part),
                       "平均和值": round(float(part.mean()), 3),
                       "最高频和值": int(sorted(counts[counts == counts.max()].index)[0]),
                       "最高频次数": int(counts.max())})
st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

st.subheader("5. 极值与区间统计")
bins = [-1, 4, 9, 14, 19, 24, 27]
labels = ["0–4", "5–9", "10–14", "15–19", "20–24", "25–27"]
range_counts = pd.cut(data["和值"], bins=bins, labels=labels).value_counts().reindex(labels, fill_value=0)
range_df = pd.DataFrame({"和值区间": labels, "次数": range_counts.to_numpy(),
                         "频率(%)": range_counts.to_numpy() / len(data) * 100})
extreme = pd.DataFrame([{"最低和值": int(data["和值"].min()), "最高和值": int(data["和值"].max()),
                         "和值标准差": float(data["和值"].std(ddof=0))}])
st.dataframe(extreme, use_container_width=True, hide_index=True)
st.dataframe(range_df, use_container_width=True, hide_index=True)

st.subheader("6. 时间顺序三模型回测对比")
backtest = evaluate_three_models(data["和值"].to_numpy(), window, strength, dynamic_window, dynamic_strength)
if backtest is None:
    st.warning("有效期数不足，无法进行时间顺序回测。建议至少 20 期；更多数据通常能提供更稳定的评估样本。")
else:
    split, detail, summary = backtest
    test_n = len(detail)
    st.write(f"按时间顺序：前 {split} 期作为初始历史，后 {test_n} 期逐期测试。每次预测只使用该期开奖前的数据。")
    if test_n < 30:
        st.warning(f"本次只有 {test_n} 期测试样本，命中率波动很大，不代表未来命中概率。")
    st.dataframe(summary.style.format({"命中率(%)": "{:.2f}", "平均和值差": "{:.3f}",
        "Log Loss（越低越好）": "{:.5f}", "Brier Score（越低越好）": "{:.5f}"}),
        use_container_width=True, hide_index=True)
    st.markdown("**逐期预测明细**")
    st.dataframe(detail, use_container_width=True, hide_index=True)

st.subheader("7. 下一期概率参考（非确定性预测）")
st.caption("公平且独立的三位 0–9 数字模型下，理论和值概率不因历史冷热变化。历史模型仅作描述性对照，不构成可靠预测。")
next_table = matrix[["和值", "理论概率(%)", "历史频率(%)", "平滑近期参考概率(%)"]].copy()
next_table["理论概率排名"] = next_table["理论概率(%)"].rank(method="min", ascending=False).astype(int)
next_table = next_table.sort_values(["理论概率(%)", "和值"], ascending=[False, True])
st.dataframe(next_table.head(10).style.format({"理论概率(%)": "{:.3f}",
    "历史频率(%)": "{:.3f}", "平滑近期参考概率(%)": "{:.3f}"}), use_container_width=True, hide_index=True)

st.subheader("8. 蒙特卡洛理论分布验证")
rng = np.random.default_rng(2026)
mc = rng.integers(0, 10, size=(mc_runs, 3)).sum(axis=1)
mc_prob = np.bincount(mc, minlength=28) / mc_runs
mc_df = pd.DataFrame({"和值": SUMS, "理论概率(%)": THEORY * 100, "蒙特卡洛概率(%)": mc_prob * 100})
with st.expander("查看蒙特卡洛分布图", expanded=False):
    st.write(f"模拟次数：{mc_runs:,}")
    st.line_chart(mc_df.set_index("和值"), use_container_width=True)

output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    data.to_excel(writer, sheet_name="清洗后历史数据", index=False)
    matrix.to_excel(writer, sheet_name="和值概率矩阵", index=False)
    digit_matrix.to_excel(writer, sheet_name="数字矩阵", index=False)
    category.to_excel(writer, sheet_name="大小单双", index=False)
    pd.DataFrame(trend_rows).to_excel(writer, sheet_name="近期动态", index=False)
    extreme.to_excel(writer, sheet_name="极值统计", index=False)
    range_df.to_excel(writer, sheet_name="区间统计", index=False)
    if backtest is not None:
        summary.to_excel(writer, sheet_name="三模型回测汇总", index=False)
        detail.to_excel(writer, sheet_name="逐期回测明细", index=False)
    mc_df.to_excel(writer, sheet_name="蒙特卡洛", index=False)
st.download_button("下载 V4.3 分析报告 Excel", output.getvalue(),
    "Canada28_Matrix_AI_V4_3_Report.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.caption("免责声明：本工具用于历史数据统计、概率教学和模型评估。随机开奖通常无法由历史数据可靠预测；模型排名不代表下一期更可能中奖，也不构成投注建议。")
