# -*- coding: utf-8 -*-
"""Canada28 Matrix AI V4.4 — live data sync + historical analysis."""
import io
import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

API_BASE = "https://pc28.world/api/v1"
DB_DIR = "data"
DB_FILE = os.path.join(DB_DIR, "canada28_history.csv")
SOURCE_NAME = "pc28.world"

st.set_page_config(page_title="Canada28 Matrix AI V4.4", page_icon="📊", layout="wide")
st.title("Canada28 Matrix AI V4.4")
st.caption("自动同步第三方公开开奖数据、历史统计与三模型时间顺序回测。随机开奖无法由历史数据可靠预测。")

COL_ALIASES = {
    "期号": ["期号", "期数", "开奖期号", "draw_id", "issue", "period", "nbr"],
    "开奖时间": ["开奖时间", "时间", "time", "draw_time"],
    "A": ["A", "号码1", "数字1", "开奖号1", "num1", "n1"],
    "B": ["B", "号码2", "数字2", "开奖号2", "num2", "n2"],
    "C": ["C", "号码3", "数字3", "开奖号3", "num3", "n3"],
}
SUMS = np.arange(28)
THEORY_COUNTS = np.bincount([a+b+c for a in range(10) for b in range(10) for c in range(10)], minlength=28)
THEORY = THEORY_COUNTS / 1000.0

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
        raise ValueError("找不到 A、B、C 三个号码列。")
    if "期号" not in out.columns:
        out.insert(0, "期号", np.arange(1, len(out)+1).astype(str))
    if "开奖时间" not in out.columns:
        out["开奖时间"] = ""
    return out[["期号", "开奖时间", "A", "B", "C"]]

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
    # API response is newest-first; all analysis uses oldest-to-newest.
    df = df.sort_values("期号", key=lambda s: pd.to_numeric(s, errors="coerce"), kind="stable")
    return df.reset_index(drop=True), original, invalid, duplicates

def api_get(path, params=None):
    response = requests.get(f"{API_BASE}{path}", params=params, timeout=20,
                            headers={"User-Agent": "Canada28-Matrix-AI/4.4"})
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok", False):
        msg = payload.get("error", {}).get("message", "API 返回异常")
        raise RuntimeError(msg)
    return payload

def normalize_api_items(payload):
    data = payload.get("data", {})
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("items", "draws", "records", "list"):
            if isinstance(data.get(key), list):
                return data[key]
        if all(k in data for k in ("nbr", "n1", "n2", "n3")):
            return [data]
    raise ValueError("接口返回格式与预期不符，未导入任何数据。")

def fetch_history(target_count=5000, progress=None):
    rows, before, seen = [], None, set()
    pages = int(np.ceil(target_count / 200))
    for page in range(pages):
        params = {"limit": min(200, target_count-len(rows))}
        if before is not None:
            params["before"] = before
        payload = api_get("/draws", params)
        items = normalize_api_items(payload)
        if not items:
            break
        for item in items:
            nbr = str(item.get("nbr", "")).strip()
            if not nbr or nbr in seen:
                continue
            try:
                a, b, c = int(item["n1"]), int(item["n2"]), int(item["n3"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(0 <= x <= 9 for x in (a, b, c)):
                continue
            if int(item.get("sum", a+b+c)) != a+b+c:
                continue
            seen.add(nbr)
            rows.append({"期号": nbr, "开奖时间": item.get("time", ""), "A": a, "B": b, "C": c})
        if progress:
            progress.progress(min(1.0, len(rows)/target_count), text=f"已读取 {len(rows):,}/{target_count:,} 期")
        if len(rows) >= target_count:
            break
        next_before = payload.get("data", {}).get("next_before") if isinstance(payload.get("data"), dict) else None
        if next_before is None:
            next_before = min((int(x["期号"]) for x in rows), default=None)
        if next_before is None or before == next_before:
            break
        before = next_before
        if len(items) < 1:
            break
        time.sleep(0.03)
    if not rows:
        raise ValueError("接口没有返回可用开奖记录。")
    return pd.DataFrame(rows)

def load_saved():
    if os.path.exists(DB_FILE):
        try:
            return pd.read_csv(DB_FILE, dtype={"期号": str}, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame(columns=["期号", "开奖时间", "A", "B", "C"])
    return pd.DataFrame(columns=["期号", "开奖时间", "A", "B", "C"])

def save_database(frame):
    os.makedirs(DB_DIR, exist_ok=True)
    frame.to_csv(DB_FILE, index=False, encoding="utf-8-sig")

def smoothed_recent_probability(history_sums, window=100, strength=100.0):
    recent = np.asarray(history_sums[-window:], dtype=int)
    counts = np.bincount(recent, minlength=28).astype(float)
    return (counts + strength * THEORY) / (len(recent) + strength)

def rolling_dynamic_probability(history_sums, window=30, strength=30.0):
    return smoothed_recent_probability(history_sums, window, strength)

def score_probabilities(probabilities, actual):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0)
    actual = np.asarray(actual, dtype=int)
    chosen = p[np.arange(len(actual)), actual]
    onehot = np.eye(28)[actual]
    return float(-np.mean(np.log(chosen))), float(np.mean(np.sum((p-onehot)**2, axis=1)))

def evaluate_three_models(sums, issue_ids, window=100, strength=100.0, dynamic_window=30, dynamic_strength=30.0):
    sums = np.asarray(sums, dtype=int)
    split = max(1, int(len(sums)*0.8))
    if len(sums)-split < 1:
        return None
    rows, probs = [], {"A 理论基准": [], "B 近期平滑": [], "C 滚动动态": []}
    for idx in range(split, len(sums)):
        hist = sums[:idx]
        model_probs = [
            ("A 理论基准", THEORY.copy()),
            ("B 近期平滑", smoothed_recent_probability(hist, window, strength)),
            ("C 滚动动态", rolling_dynamic_probability(hist, dynamic_window, dynamic_strength)),
        ]
        actual = int(sums[idx])
        row = {"期号": str(issue_ids[idx]), "预测前已知历史期数": idx, "实际和值": actual}
        for name, p in model_probs:
            pred = int(np.flatnonzero(np.isclose(p, p.max()))[0])
            row[f"{name}预测和值"] = pred
            row[f"{name}是否命中"] = "命中" if pred == actual else "未命中"
            row[f"{name}和值差"] = abs(pred-actual)
            probs[name].append(p)
        rows.append(row)
    detail = pd.DataFrame(rows)
    actual = sums[split:]
    summary = []
    for name in probs:
        hits = int((detail[f"{name}是否命中"] == "命中").sum())
        ll, bs = score_probabilities(probs[name], actual)
        summary.append({"模型": name, "完全命中": hits, "测试期数": len(actual),
                        "命中率(%)": hits/len(actual)*100,
                        "平均和值差": float(detail[f"{name}和值差"].mean()),
                        "Log Loss（越低越好）": ll, "Brier Score（越低越好）": bs})
    return split, detail, pd.DataFrame(summary)

# Sidebar controls
st.sidebar.header("数据中心")
source_mode = st.sidebar.radio("数据来源", ["自动同步官网数据", "上传 Excel / CSV"])
history_count = st.sidebar.selectbox("首次导入历史期数", [1000, 3000, 5000], index=2)
window = st.sidebar.select_slider("近期统计窗口", options=[20, 50, 100, 200, 300, 500], value=100)
strength = st.sidebar.slider("近期频率平滑强度", 10, 500, 100, 10)
dynamic_window = st.sidebar.select_slider("动态模型滚动窗口", options=[10, 20, 30, 50, 100], value=30)
dynamic_strength = st.sidebar.slider("动态模型平滑强度", 5, 200, 30, 5)
mc_runs = st.sidebar.selectbox("蒙特卡洛模拟次数", [100_000, 500_000, 1_000_000], index=0)

if "api_history" not in st.session_state:
    st.session_state.api_history = load_saved()

if source_mode == "自动同步官网数据":
    st.markdown("### 1. 自动开奖数据同步")
    st.caption("数据来源：pc28.world 第三方公开只读 API；请以官方开奖公告为准。")
    auto_update = st.checkbox("开启自动检查新开奖（网页保持打开时，每 3.5 分钟检查一次）", value=True)
    refresh_count = st_autorefresh(interval=210_000, key="pc28_auto_refresh") if auto_update else 0
    if auto_update and refresh_count > 0 and st.session_state.get("last_auto_refresh_count") != refresh_count:
        try:
            latest_payload = api_get("/draws/latest")
            latest_data = latest_payload.get("data", {})
            latest_draw = latest_data.get("draw", latest_data) if isinstance(latest_data, dict) else {}
            if all(k in latest_draw for k in ("nbr", "n1", "n2", "n3")):
                latest_row = pd.DataFrame([{
                    "期号": str(latest_draw["nbr"]), "开奖时间": latest_draw.get("time", ""),
                    "A": latest_draw["n1"], "B": latest_draw["n2"], "C": latest_draw["n3"]
                }])
                combined = pd.concat([st.session_state.api_history, latest_row], ignore_index=True)
                combined, _, _, _ = clean_data(combined)
                st.session_state.api_history = combined
                save_database(combined)
                st.session_state.last_auto_refresh_count = refresh_count
                st.toast(f"自动检查完成：最新期号 {latest_row.iloc[0]['期号']}")
            else:
                st.session_state.last_auto_refresh_count = refresh_count
                st.warning("自动检查已执行，但接口未返回可识别的最新开奖字段。")
        except Exception as exc:
            st.session_state.last_auto_refresh_count = refresh_count
            st.warning(f"自动检查暂时失败：{exc}")
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("首次导入 / 补足历史数据", type="primary", use_container_width=True):
            bar = st.progress(0, text="正在连接数据接口…")
            try:
                fetched = fetch_history(history_count, bar)
                st.session_state.api_history = fetched
                save_database(fetched)
                st.success(f"成功获取并保存 {len(fetched):,} 期记录。")
                st.rerun()
            except Exception as exc:
                st.error(f"同步失败：{exc}")
    with c2:
        if st.button("同步最新开奖", use_container_width=True):
            try:
                payload = api_get("/draws", {"limit": 200})
                fresh = pd.DataFrame(normalize_api_items(payload))
                fresh = fresh.rename(columns={"nbr":"期号", "time":"开奖时间", "n1":"A", "n2":"B", "n3":"C"})
                fresh = fresh[["期号","开奖时间","A","B","C"]]
                old = st.session_state.api_history
                combined = pd.concat([old, fresh], ignore_index=True)
                combined, _, _, _ = clean_data(combined)
                st.session_state.api_history = combined
                save_database(combined)
                st.success(f"同步完成，当前累计 {len(combined):,} 期。")
                st.rerun()
            except Exception as exc:
                st.error(f"最新数据同步失败：{exc}")
    db = st.session_state.api_history
    if not db.empty:
        st.info(f"当前本地数据：{len(db):,} 期。最新期号：{db.iloc[-1]['期号']}。")
        st.download_button("下载本地历史数据 CSV 备份", db.to_csv(index=False).encode("utf-8-sig"),
                           "Canada28_history_backup.csv", "text/csv")
    else:
        st.warning("尚未导入历史数据。请点击「首次导入 / 补足历史数据」。")
    raw = db
else:
    upload = st.sidebar.file_uploader("上传真实历史开奖数据（Excel / CSV）", type=["xlsx","xls","csv"])
    if upload is None:
        st.info("请在左侧上传 Excel 或 CSV，或切换到自动同步官网数据。")
        st.stop()
    content = upload.getvalue()
    if upload.name.lower().endswith((".xlsx",".xls")):
        raw = pd.read_excel(io.BytesIO(content))
    else:
        try:
            raw = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = pd.read_csv(io.BytesIO(content), encoding="gbk")

if raw is None or raw.empty:
    st.stop()

try:
    data, original_rows, invalid_rows, duplicate_rows = clean_data(raw)
except Exception as exc:
    st.error(f"数据读取或格式检查失败：{exc}")
    st.stop()
if data.empty:
    st.error("没有可分析的有效记录。")
    st.stop()

st.success(f"成功载入 {len(data):,} 期有效记录。分析顺序已统一为旧 → 新。")
m1,m2,m3,m4 = st.columns(4)
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
matrix = pd.DataFrame({"和值":SUMS, "理论概率(%)":THEORY*100,
                       "历史频率(%)":full_counts.to_numpy()/len(data)*100,
                       f"近{len(recent)}期频率(%)":recent_counts.to_numpy()/len(recent)*100,
                       "平滑近期参考概率(%)":recent_prob*100})
st.subheader("2. 0–27 和值概率矩阵")
st.dataframe(matrix.style.format({c:"{:.3f}" for c in matrix.columns if c!="和值"}), use_container_width=True, hide_index=True)
st.line_chart(matrix.set_index("和值")[["理论概率(%)","历史频率(%)","平滑近期参考概率(%)"]], use_container_width=True)

c1,c2 = st.columns(2)
with c1:
    st.subheader("3. A / B / C 数字分布")
    digit_matrix = pd.DataFrame({"数字":range(10)})
    for col in ["A","B","C"]:
        counts = data[col].value_counts().reindex(range(10), fill_value=0)
        digit_matrix[f"{col}次数"] = counts.to_numpy()
        digit_matrix[f"{col}频率(%)"] = counts.to_numpy()/len(data)*100
    st.dataframe(digit_matrix, use_container_width=True, hide_index=True)
with c2:
    st.subheader("4. 大小 / 单双统计")
    category = pd.DataFrame({"类别":["大(14–27)","小(0–13)","单","双"],
        "次数":[int((data["和值"]>=14).sum()),int((data["和值"]<=13).sum()),
               int((data["和值"]%2==1).sum()),int((data["和值"]%2==0).sum())]})
    category["频率(%)"] = category["次数"]/len(data)*100
    st.dataframe(category, use_container_width=True, hide_index=True)

st.subheader("5. 近期动态统计（20 / 50 / 100 期）")
trend_rows=[]
for n in [20,50,100]:
    part=data["和值"].tail(min(n,len(data)))
    counts=part.value_counts()
    trend_rows.append({"统计窗口":f"近{len(part)}期","样本数":len(part),"平均和值":round(float(part.mean()),3),
                       "最高频和值":int(sorted(counts[counts==counts.max()].index)[0]),"最高频次数":int(counts.max())})
st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

st.subheader("6. 极值与区间统计")
bins=[-1,4,9,14,19,24,27]
labels=["0–4","5–9","10–14","15–19","20–24","25–27"]
range_counts=pd.cut(data["和值"],bins=bins,labels=labels).value_counts().reindex(labels,fill_value=0)
range_df=pd.DataFrame({"和值区间":labels,"次数":range_counts.to_numpy(),"频率(%)":range_counts.to_numpy()/len(data)*100})
extreme=pd.DataFrame([{"最低和值":int(data["和值"].min()),"最高和值":int(data["和值"].max()),
                       "和值标准差":float(data["和值"].std(ddof=0))}])
st.dataframe(extreme,use_container_width=True,hide_index=True)
st.dataframe(range_df,use_container_width=True,hide_index=True)

st.subheader("7. 时间顺序三模型回测对比")
backtest=evaluate_three_models(data["和值"].to_numpy(),data["期号"].astype(str).to_numpy(),window,strength,dynamic_window,dynamic_strength)
if backtest is None:
    st.warning("有效期数不足，无法进行时间顺序回测。")
else:
    split,detail,summary=backtest
    test_n=len(detail)
    st.write(f"前 {split} 期作为初始历史，后 {test_n} 期逐期测试；每次只使用开奖前的数据。")
    if test_n<30:
        st.warning(f"本次只有 {test_n} 期测试样本，命中率波动很大，不代表未来命中概率。")
    st.dataframe(summary.style.format({"命中率(%)":"{:.2f}","平均和值差":"{:.3f}",
        "Log Loss（越低越好）":"{:.5f}","Brier Score（越低越好）":"{:.5f}"}),use_container_width=True,hide_index=True)
    st.markdown("**逐期预测明细**")
    st.dataframe(detail,use_container_width=True,hide_index=True)

st.subheader("8. 下一期概率参考（非确定性预测）")
st.caption("公平独立的三位 0–9 数字模型下，理论和值概率不因历史冷热变化。")
next_table=matrix[["和值","理论概率(%)","历史频率(%)","平滑近期参考概率(%)"]].copy()
next_table["理论概率排名"]=next_table["理论概率(%)"].rank(method="min",ascending=False).astype(int)
next_table=next_table.sort_values(["理论概率(%)","和值"],ascending=[False,True])
st.dataframe(next_table.head(10).style.format({"理论概率(%)":"{:.3f}","历史频率(%)":"{:.3f}","平滑近期参考概率(%)":"{:.3f}"}),use_container_width=True,hide_index=True)

st.subheader("9. 蒙特卡洛理论分布验证")
rng=np.random.default_rng(2026)
mc=rng.integers(0,10,size=(mc_runs,3)).sum(axis=1)
mc_prob=np.bincount(mc,minlength=28)/mc_runs
mc_df=pd.DataFrame({"和值":SUMS,"理论概率(%)":THEORY*100,"蒙特卡洛概率(%)":mc_prob*100})
with st.expander("查看蒙特卡洛分布图",expanded=False):
    st.write(f"模拟次数：{mc_runs:,}")
    st.line_chart(mc_df.set_index("和值"),use_container_width=True)

output=io.BytesIO()
with pd.ExcelWriter(output,engine="openpyxl") as writer:
    data.to_excel(writer,sheet_name="清洗后历史数据",index=False)
    matrix.to_excel(writer,sheet_name="和值概率矩阵",index=False)
    digit_matrix.to_excel(writer,sheet_name="数字矩阵",index=False)
    category.to_excel(writer,sheet_name="大小单双",index=False)
    pd.DataFrame(trend_rows).to_excel(writer,sheet_name="近期动态",index=False)
    extreme.to_excel(writer,sheet_name="极值统计",index=False)
    range_df.to_excel(writer,sheet_name="区间统计",index=False)
    if backtest is not None:
        summary.to_excel(writer,sheet_name="三模型回测汇总",index=False)
        detail.to_excel(writer,sheet_name="逐期回测明细",index=False)
    mc_df.to_excel(writer,sheet_name="蒙特卡洛",index=False)
st.download_button("下载 V4.4 分析报告 Excel",output.getvalue(),"Canada28_Matrix_AI_V4_4_Report.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.caption("数据来源：pc28.world（第三方公开数据接口，非 BCLC 官方网站）；请核对官方公告。免责声明：随机开奖通常无法由历史数据可靠预测，模型排名不代表下一期更可能中奖，也不构成投注建议。")
