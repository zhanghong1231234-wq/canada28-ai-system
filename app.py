# -*- coding: utf-8 -*-
"""Canada28 Matrix AI V4.0 — historical analysis and probability backtesting."""
import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Canada28 Matrix AI V4.1", page_icon="📊", layout="wide")

st.title("Canada28 Matrix AI V4.1")
st.caption("真实历史数据分析、动态窗口矩阵、极值统计与时间顺序回测。统计结果不保证预测未来开奖结果。")

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
    # Bayesian-style shrinkage toward the exact three-digit theoretical distribution.
    return (counts + strength * THEORY) / (len(recent) + strength)

def evaluate_walk_forward(sums, window=100, strength=100.0):
    # First 80% is the initial training segment; remaining rows are predicted sequentially.
    split = max(1, int(len(sums) * 0.8))
    if len(sums) - split < 1:
        return None
    actual = np.asarray(sums[split:], dtype=int)
    p_theory, p_recent = [], []
    for idx in range(split, len(sums)):
        hist = np.asarray(sums[:idx], dtype=int)
        p_theory.append(THEORY.copy())
        p_recent.append(smoothed_recent_probability(hist, window, strength))
    def scores(probabilities):
        p = np.clip(np.asarray(probabilities), 1e-15, 1.0)
        chosen = p[np.arange(len(actual)), actual]
        onehot = np.eye(28)[actual]
        return float(-np.mean(np.log(chosen))), float(np.mean(np.sum((p-onehot)**2, axis=1)))
    return split, len(actual), scores(p_theory), scores(p_recent)

upload = st.sidebar.file_uploader("上传真实历史开奖数据（Excel / CSV）", type=["xlsx", "xls", "csv"])
st.sidebar.markdown("数据至少包含 A、B、C 三列；也支持「号码1、号码2、号码3」。")
window = st.sidebar.select_slider("平滑参考窗口", options=[20, 50, 100, 200, 300, 500], value=100)
strength = st.sidebar.slider("近期频率平滑强度", min_value=10, max_value=500, value=100, step=10)
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

# Matrix table: all 28 sums, exact theoretical probability, full-history frequency, recent frequency.
full_counts = data["和值"].value_counts().reindex(SUMS, fill_value=0)
recent = data["和值"].tail(window)
recent_counts = recent.value_counts().reindex(SUMS, fill_value=0)
recent_prob = smoothed_recent_probability(data["和值"].to_numpy(), window, strength)
matrix = pd.DataFrame({
    "和值": SUMS,
    "理论概率(%)": THEORY * 100,
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
        "次数": [
            int((data["和值"] >= 14).sum()), int((data["和值"] <= 13).sum()),
            int((data["和值"] % 2 == 1).sum()), int((data["和值"] % 2 == 0).sum())
        ],
    })
    category["频率(%)"] = category["次数"] / len(data) * 100
    st.dataframe(category, use_container_width=True, hide_index=True)

st.subheader("4. 近期动态统计（20 / 50 / 100 期）")
window_rows = []
for n in [20, 50, 100]:
    part = data.tail(min(n, len(data)))
    sums_part = part["和值"]
    window_rows.append({
        "统计窗口": f"最近{len(part)}期",
        "平均和值": float(sums_part.mean()),
        "最大和值": int(sums_part.max()),
        "最小和值": int(sums_part.min()),
        "大(14–27)%": float((sums_part >= 14).mean() * 100),
        "小(0–13)%": float((sums_part <= 13).mean() * 100),
        "单数%": float((sums_part % 2 == 1).mean() * 100),
        "双数%": float((sums_part % 2 == 0).mean() * 100),
    })
window_stats = pd.DataFrame(window_rows)
st.dataframe(window_stats.style.format({c: "{:.2f}" for c in window_stats.columns if c not in ["统计窗口"]}),
             use_container_width=True, hide_index=True)

extreme_counts = pd.DataFrame({
    "区间": ["极小和值(0–6)", "中间和值(7–20)", "极大和值(21–27)"],
    "次数": [
        int(data["和值"].between(0, 6).sum()),
        int(data["和值"].between(7, 20).sum()),
        int(data["和值"].between(21, 27).sum())
    ]
})
extreme_counts["频率(%)"] = extreme_counts["次数"] / len(data) * 100
st.subheader("5. 极值与区间统计")
st.dataframe(extreme_counts, use_container_width=True, hide_index=True)

st.subheader("6. 时间顺序模型回测")
backtest = evaluate_walk_forward(data["和值"].to_numpy(), window, strength)
if backtest is None:
    st.warning("至少需要 10 期以上数据，才能进行有意义的时间顺序回测。")
else:
    split, test_n, base, recent_score = backtest
    result = pd.DataFrame({
        "模型": ["理论概率基准", f"近{window}期频率（向理论概率平滑）"],
        "Log Loss（越低越好）": [base[0], recent_score[0]],
        "Brier Score（越低越好）": [base[1], recent_score[1]],
    })
    st.write(f"按时间顺序：前 {split} 期作为初始历史，后 {test_n} 期逐期回测。")
    st.dataframe(result.style.format({"Log Loss（越低越好）": "{:.5f}", "Brier Score（越低越好）": "{:.5f}"}),
                 use_container_width=True, hide_index=True)

st.subheader("7. 下一期概率参考（非确定性预测）")
st.caption("公平且独立的三位 0–9 数字模型下，理论和值概率不因历史冷热变化。近期平滑概率仅作为样本描述对照。")
next_table = matrix[["和值", "理论概率(%)", "历史频率(%)", "平滑近期参考概率(%)"]].copy()
next_table["理论概率排名"] = next_table["理论概率(%)"].rank(method="min", ascending=False).astype(int)
next_table = next_table.sort_values(["理论概率(%)", "和值"], ascending=[False, True])
st.dataframe(next_table.head(10).style.format({
    "理论概率(%)": "{:.3f}", "历史频率(%)": "{:.3f}", "平滑近期参考概率(%)": "{:.3f}"
}), use_container_width=True, hide_index=True)

# Monte Carlo validation of theoretical distribution.
rng = np.random.default_rng(2026)
mc = rng.integers(0, 10, size=(mc_runs, 3)).sum(axis=1)
mc_prob = np.bincount(mc, minlength=28) / mc_runs
mc_df = pd.DataFrame({"和值": SUMS, "理论概率(%)": THEORY * 100, "蒙特卡洛概率(%)": mc_prob * 100})
with st.expander("8. 蒙特卡洛理论分布验证"):
    st.write(f"模拟次数：{mc_runs:,}")
    st.line_chart(mc_df.set_index("和值"), use_container_width=True)

# Export workbook with clean data, matrix, backtest, digit distributions.
output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    data.to_excel(writer, sheet_name="清洗后历史数据", index=False)
    matrix.to_excel(writer, sheet_name="和值概率矩阵", index=False)
    digit_matrix.to_excel(writer, sheet_name="数字矩阵", index=False)
    category.to_excel(writer, sheet_name="大小单双", index=False)
    window_stats.to_excel(writer, sheet_name="近期动态统计", index=False)
    extreme_counts.to_excel(writer, sheet_name="极值区间统计", index=False)
    if backtest is not None:
        result.to_excel(writer, sheet_name="模型回测", index=False)
    mc_df.to_excel(writer, sheet_name="蒙特卡洛", index=False)
st.download_button("下载 V4.1 分析报告 Excel", output.getvalue(),
                   "Canada28_Matrix_AI_V4_1_Report.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("免责声明：本工具用于历史数据统计、概率教学和模型评估。随机开奖通常无法由历史数据可靠预测；排名不代表下一期更可能中奖，也不构成投注建议。")
