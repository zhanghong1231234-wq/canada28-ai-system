# -*- coding: utf-8 -*-
import io
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="加拿大 PC28 数据分析平台",
    page_icon="📊",
    layout="wide",
)

REQUIRED_COLUMNS = ["期号", "号码1", "号码2", "号码3"]

st.title("加拿大 PC28 数据分析平台")
st.caption("历史数据统计与可视化工具；统计结果仅描述历史数据，不代表未来结果。")

@st.cache_data
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    lower_name = filename.lower()
    if lower_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    try:
        return pd.read_csv(buffer, encoding="utf-8-sig")
    except UnicodeDecodeError:
        buffer.seek(0)
        return pd.read_csv(buffer, encoding="gbk")

uploaded_file = st.sidebar.file_uploader(
    "上传历史数据（CSV 或 Excel）",
    type=["csv", "xlsx", "xls"],
)

st.sidebar.markdown(
    "数据表必须包含：`期号`、`号码1`、`号码2`、`号码3`。"
)

if uploaded_file is None:
    st.info("请先在左侧上传 CSV 或 Excel 历史数据文件。")
    template = pd.DataFrame(columns=REQUIRED_COLUMNS)
    st.download_button(
        "下载空白 CSV 模板",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="pc28_data_template.csv",
        mime="text/csv",
    )
    st.stop()

try:
    raw = load_data(uploaded_file.getvalue(), uploaded_file.name)
except Exception as exc:
    st.error(f"文件读取失败：{exc}")
    st.stop()

missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
if missing:
    st.error("文件缺少必要列：" + "、".join(missing))
    st.write("当前文件列名：", list(raw.columns))
    st.stop()

data = raw[REQUIRED_COLUMNS].copy()
data["期号"] = data["期号"].astype(str).str.strip()

for column in ["号码1", "号码2", "号码3"]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data = data.dropna(subset=REQUIRED_COLUMNS).copy()
valid_rows = data[["号码1", "号码2", "号码3"]].apply(
    lambda col: col.between(0, 9)
).all(axis=1)
invalid_count = int((~valid_rows).sum())
data = data[valid_rows].copy()

data = data.drop_duplicates(subset=["期号"], keep="last")
for column in ["号码1", "号码2", "号码3"]:
    data[column] = data[column].astype(int)

if data.empty:
    st.warning("没有可分析的有效记录，请检查文件内容。")
    st.stop()

data["和值"] = data[["号码1", "号码2", "号码3"]].sum(axis=1)
data["大小"] = data["和值"].apply(lambda value: "大" if value >= 14 else "小")
data["单双"] = data["和值"].apply(lambda value: "单" if value % 2 else "双")

st.success(f"成功读取 {len(data)} 条有效记录。")
if invalid_count:
    st.warning(f"已排除 {invalid_count} 条号码不在 0–9 范围内的记录。")

col1, col2, col3, col4 = st.columns(4)
col1.metric("有效期数", f"{len(data):,}")
col2.metric("平均和值", f"{data['和值'].mean():.2f}")
col3.metric("最高和值", int(data["和值"].max()))
col4.metric("最低和值", int(data["和值"].min()))

st.subheader("最近 20 期数据")
st.dataframe(data.tail(20), use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.subheader("和值历史走势")
    chart_data = data[["期号", "和值"]].tail(300).set_index("期号")
    st.line_chart(chart_data, use_container_width=True)

with right:
    st.subheader("和值出现次数")
    sum_counts = (
        data["和值"].value_counts()
        .reindex(range(28), fill_value=0)
        .rename_axis("和值")
        .rename("次数")
    )
    st.bar_chart(sum_counts, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("大小分布")
    size_counts = data["大小"].value_counts().reindex(["大", "小"], fill_value=0)
    st.bar_chart(size_counts, use_container_width=True)
    st.dataframe(size_counts.rename("次数"), use_container_width=True)

with right:
    st.subheader("单双分布")
    parity_counts = data["单双"].value_counts().reindex(["单", "双"], fill_value=0)
    st.bar_chart(parity_counts, use_container_width=True)
    st.dataframe(parity_counts.rename("次数"), use_container_width=True)

st.subheader("数字 0–9 出现次数")
digit_counts = (
    pd.concat([data["号码1"], data["号码2"], data["号码3"]])
    .value_counts()
    .reindex(range(10), fill_value=0)
    .rename_axis("数字")
    .rename("次数")
)
st.bar_chart(digit_counts, use_container_width=True)

st.subheader("和值描述统计")
st.dataframe(data["和值"].describe().to_frame("统计值"), use_container_width=True)

csv_bytes = data.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "下载清洗后的分析数据",
    data=csv_bytes,
    file_name="pc28_analysis_cleaned.csv",
    mime="text/csv",
)

st.caption(
    "说明：大小按和值 ≥ 14 记为“大”，否则为“小”；单双按和值奇偶划分。"
    "历史频率和图表仅用于描述样本，不构成预测、投注建议或收益保证。"
)
