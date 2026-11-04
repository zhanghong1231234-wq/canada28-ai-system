# -*- coding: utf-8 -*-
"""Canada28 Matrix AI V4.5 — adaptive walk-forward validation."""
import io
import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

API_BASE = "https://pc28.world/api/v1"
DB_DIR = "data"
DB_FILE = os.path.join(DB_DIR, "canada28_history.csv")

st.set_page_config(page_title="Canada28 Matrix AI V4.5", page_icon="📊", layout="wide")
st.title("Canada28 Matrix AI V4.5")
st.caption("自适应参数验证、严格时间顺序测试与历史统计。随机开奖无法由历史数据可靠预测。")

ALIASES = {
    "期号": ["期号", "期数", "开奖期号", "draw_id", "issue", "period", "nbr"],
    "开奖时间": ["开奖时间", "时间", "time", "draw_time"],
    "A": ["A", "号码1", "数字1", "开奖号1", "num1", "n1"],
    "B": ["B", "号码2", "数字2", "开奖号2", "num2", "n2"],
    "C": ["C", "号码3", "数字3", "开奖号3", "num3", "n3"],
}
SUMS = np.arange(28)
THEORY_COUNTS = np.bincount(
    [a + b + c for a in range(10) for b in range(10) for c in range(10)],
    minlength=28,
)
THEORY = THEORY_COUNTS / 1000.0
EMPTY = pd.DataFrame(columns=["期号", "开奖时间", "A", "B", "C"])


def standardize_columns(frame):
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    rename = {}
    for target, names in ALIASES.items():
        found = next((lookup[x.lower()] for x in names if x.lower() in lookup), None)
        if found is not None:
            rename[found] = target
    out = frame.rename(columns=rename).copy()
    if not all(c in out.columns for c in ["A", "B", "C"]):
        raise ValueError("找不到 A、B、C 三个号码列。")
    if "期号" not in out.columns:
        out.insert(0, "期号", np.arange(1, len(out) + 1).astype(str))
    if "开奖时间" not in out.columns:
        out["开奖时间"] = ""
    return out[["期号", "开奖时间", "A", "B", "C"]]


def clean_data(raw):
    df = standardize_columns(raw)
    df["期号"] = df["期号"].astype(str).str.strip()
    for c in ["A", "B", "C"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    original = len(df)
    df = df.dropna(subset=["期号", "A", "B", "C"]).copy()
    valid = df[["A", "B", "C"]].apply(lambda s: s.between(0, 9) & (s % 1 == 0)).all(axis=1)
    invalid = int((~valid).sum())
    df = df[valid].copy()
    duplicates = int(df.duplicated(subset=["期号"], keep="last").sum())
    df = df.drop_duplicates(subset=["期号"], keep="last")
    for c in ["A", "B", "C"]:
        df[c] = df[c].astype(int)
    df["和值"] = df[["A", "B", "C"]].sum(axis=1)
    df["大小"] = np.where(df["和值"] >= 14, "大", "小")
    df["单双"] = np.where(df["和值"] % 2 == 0, "双", "单")
    issue_num = pd.to_numeric(df["期号"], errors="coerce")
    if issue_num.notna().all():
        df["_order"] = issue_num
        df = df.sort_values("_order", kind="stable").drop(columns="_order")
    return df.reset_index(drop=True), original, invalid, duplicates


def api_get(path, params=None):
    r = requests.get(API_BASE + path, params=params, timeout=25,
                     headers={"User-Agent": "Canada28-Matrix-AI/4.5"})
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, dict) or not payload.get("ok", False):
        detail = payload.get("error", {}) if isinstance(payload, dict) else {}
        message = detail.get("message", "API 返回异常") if isinstance(detail, dict) else str(detail)
        raise RuntimeError(message)
    return payload


def api_items(payload):
    data = payload.get("data", {})
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("items", "draws", "records", "list"):
            if isinstance(data.get(k), list):
                return data[k]
        if all(k in data for k in ("nbr", "n1", "n2", "n3")):
            return [data]
    raise ValueError("接口返回格式与预期不符。")


def fetch_history(count, progress=None):
    rows, seen, before = [], set(), None
    for _ in range(int(np.ceil(count / 200))):
        params = {"limit": min(200, count - len(rows))}
        if before is not None:
            params["before"] = before
        payload = api_get("/draws", params)
        items = api_items(payload)
        if not items:
            break
        for x in items:
            issue = str(x.get("nbr", "")).strip()
            if not issue or issue in seen:
                continue
            try:
                a, b, c = int(x["n1"]), int(x["n2"]), int(x["n3"])
                if not all(0 <= v <= 9 for v in (a, b, c)):
                    continue
                if int(x.get("sum", a+b+c)) != a+b+c:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            seen.add(issue)
            rows.append({"期号": issue, "开奖时间": x.get("time", ""), "A": a, "B": b, "C": c})
        if progress:
            progress.progress(min(1.0, len(rows)/count), text=f"已读取 {len(rows):,}/{count:,} 期")
        if len(rows) >= count:
            break
        d = payload.get("data", {})
        nxt = d.get("next_before") if isinstance(d, dict) else None
        if nxt is None:
            ids = []
            for x in rows:
                try: ids.append(int(x["期号"]))
                except ValueError: pass
            nxt = min(ids) if ids else None
        if nxt is None or str(nxt) == str(before):
            break
        before = nxt
        time.sleep(0.03)
    if not rows:
        raise ValueError("接口没有返回可用开奖记录。")
    result, _, _, _ = clean_data(pd.DataFrame(rows))
    return result.tail(count).reset_index(drop=True)


def load_saved():
    if os.path.exists(DB_FILE):
        try:
            return pd.read_csv(DB_FILE, dtype={"期号": str}, encoding="utf-8-sig")
        except Exception:
            pass
    return EMPTY.copy()


def save_db(df):
    os.makedirs(DB_DIR, exist_ok=True)
    df.to_csv(DB_FILE, index=False, encoding="utf-8-sig")


def smoothed_prob(hist, window, strength):
    sample = np.asarray(hist[-window:], dtype=int)
    counts = np.bincount(sample, minlength=28).astype(float)
    return (counts + strength * THEORY) / (len(sample) + strength)


def loss_metrics(probabilities, actual):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0)
    y = np.asarray(actual, dtype=int)
    onehot = np.eye(28)[y]
    return float(-np.mean(np.log(p[np.arange(len(y)), y]))), float(np.mean(np.sum((p-onehot)**2, axis=1)))


def predict_sequence(history, targets, window, strength):
    """每个目标点只使用其之前的历史数据。"""
    rows = []
    for idx in targets:
        p = smoothed_prob(history[:idx], window, strength)
        rows.append(p)
    return np.asarray(rows)


def evaluate_param(history, targets, actual, window, strength):
    probs = predict_sequence(history, targets, window, strength)
    ll, bs = loss_metrics(probs, actual)
    return ll, bs, probs


def adaptive_walk_forward(sums, issues):
    """60% 初始训练、20% 参数验证、20% 最终留出测试。"""
    n = len(sums)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    if train_end < 100 or val_end <= train_end or n - val_end < 30:
        return None

    # 参数只在验证段选择；最终测试段不参与调参。
    candidates = [(w, s) for w in [30, 50, 100, 200, 500, 1000]
                  for s in [20, 50, 100, 200]]
    val_targets = np.arange(train_end, val_end)
    val_actual = sums[val_targets]
    scored = []
    for w, strength in candidates:
        ll, bs, _ = evaluate_param(sums, val_targets, val_actual, w, strength)
        scored.append({"窗口": w, "平滑强度": strength, "验证Log Loss": ll,
                       "验证Brier": bs})
    tuning = pd.DataFrame(scored).sort_values(
        ["验证Log Loss", "验证Brier"], ascending=True
    ).reset_index(drop=True)
    best = tuning.iloc[0]
    best_w, best_s = int(best["窗口"]), float(best["平滑强度"])

    test_targets = np.arange(val_end, n)
    test_actual = sums[test_targets]
    theory_probs = np.tile(THEORY, (len(test_targets), 1))
    adaptive_probs = predict_sequence(sums, test_targets, best_w, best_s)
    # 传统近期窗口作为固定参照，参数不从测试段选择。
    fixed_probs = predict_sequence(sums, test_targets, 100, 100)
    models = {
        "A 理论基准": theory_probs,
        "B 固定近期平滑(100/100)": fixed_probs,
        f"C 自适应平滑({best_w}/{best_s:g})": adaptive_probs,
    }
    summary_rows, detail = [], pd.DataFrame({"期号": np.asarray(issues)[test_targets],
                                             "实际和值": test_actual})
    for name, probs in models.items():
        pred = probs.argmax(axis=1)
        hit = pred == test_actual
        ll, bs = loss_metrics(probs, test_actual)
        summary_rows.append({
            "模型": name, "完全命中": int(hit.sum()), "测试期数": len(test_actual),
            "命中率(%)": float(hit.mean()*100),
            "平均和值差": float(np.mean(np.abs(pred-test_actual))),
            "Log Loss（越低越好）": ll, "Brier Score（越低越好）": bs,
        })
        detail[name+"预测和值"] = pred
        detail[name+"是否命中"] = np.where(hit, "命中", "未命中")
        detail[name+"和值差"] = np.abs(pred-test_actual)
    # 配对逐期 Log Loss 差异及标准误，作为描述性不确定性参考。
    base_loss = -np.log(np.clip(theory_probs[np.arange(len(test_actual)), test_actual], 1e-15, 1))
    comparison = []
    for name, probs in list(models.items())[1:]:
        model_loss = -np.log(np.clip(probs[np.arange(len(test_actual)), test_actual], 1e-15, 1))
        diff = model_loss - base_loss
        se = float(np.std(diff, ddof=1)/np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
        comparison.append({"模型": name, "相对理论基准平均Log Loss差": float(np.mean(diff)),
                           "差值标准误": se,
                           "描述性结论": "低于理论基准" if np.mean(diff) < 0 else "未低于理论基准"})
    return train_end, val_end, tuning, pd.DataFrame(summary_rows), detail, pd.DataFrame(comparison), best_w, best_s


# Sidebar and data source
st.sidebar.header("数据中心")
mode = st.sidebar.radio("数据来源", ["自动同步第三方数据", "上传 Excel / CSV"])
count = st.sidebar.selectbox("首次导入历史期数", [1000, 3000, 5000], index=2)
if "history" not in st.session_state:
    st.session_state.history = load_saved()

if mode == "自动同步第三方数据":
    st.caption("数据来源：pc28.world 第三方公开只读 API，非 BCLC 官方网站；请核对官方开奖公告。")
    auto = st.checkbox("开启自动检查新开奖（网页保持打开时，每 3.5 分钟）", True)
    @st.fragment(run_every="210s" if auto else None)
    def refresh_latest():
        if not auto:
            return
        try:
            p = api_get("/draws/latest")
            d = p.get("data", {})
            x = d.get("draw", d) if isinstance(d, dict) else {}
            if not all(k in x for k in ("nbr", "n1", "n2", "n3")):
                return
            new = pd.DataFrame([{"期号": str(x["nbr"]), "开奖时间": x.get("time", ""),
                                 "A": x["n1"], "B": x["n2"], "C": x["n3"]}])
            merged, _, _, _ = clean_data(pd.concat([st.session_state.history, new], ignore_index=True))
            st.session_state.history = merged
            save_db(merged)
        except Exception as e:
            st.warning(f"自动检查失败：{e}")
    refresh_latest()
    left, right = st.columns(2)
    with left:
        if st.button("首次导入 / 补足历史数据", type="primary", use_container_width=True):
            bar = st.progress(0)
            try:
                df = fetch_history(count, bar)
                st.session_state.history = df
                save_db(df)
                st.success(f"已导入 {len(df):,} 期。")
                st.rerun()
            except Exception as e:
                st.error(f"同步失败：{e}")
    with right:
        if st.button("同步最新开奖", use_container_width=True):
            try:
                p = api_get("/draws", {"limit": 200})
                rows = api_items(p)
                fresh = pd.DataFrame(rows).rename(columns={"nbr":"期号","time":"开奖时间","n1":"A","n2":"B","n3":"C"})
                fresh = fresh[["期号","开奖时间","A","B","C"]]
                merged, _, _, _ = clean_data(pd.concat([st.session_state.history, fresh], ignore_index=True))
                st.session_state.history = merged
                save_db(merged)
                st.success(f"同步完成，共 {len(merged):,} 期。")
                st.rerun()
            except Exception as e:
                st.error(f"同步失败：{e}")
    raw = st.session_state.history
    if not raw.empty:
        st.info(f"本地记录 {len(raw):,} 期；最新期号 {raw.iloc[-1]['期号']}。")
        st.download_button("下载历史数据 CSV 备份", raw.to_csv(index=False).encode("utf-8-sig"),
                           "Canada28_history_backup.csv", "text/csv")
    else:
        st.warning("尚无历史数据，请先导入。")
else:
    upload = st.sidebar.file_uploader("上传 Excel / CSV", type=["xlsx","xls","csv"])
    if upload is None:
        st.info("请上传历史开奖文件，或切换到自动同步数据。")
        st.stop()
    content = upload.getvalue()
    raw = pd.read_excel(io.BytesIO(content)) if upload.name.lower().endswith((".xlsx",".xls")) else pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")

if raw is None or raw.empty:
    st.stop()
try:
    data, original, invalid, duplicates = clean_data(raw)
except Exception as e:
    st.error(f"数据格式错误：{e}")
    st.stop()
if data.empty:
    st.error("没有可分析的有效数据。")
    st.stop()

st.success(f"载入 {len(data):,} 期有效记录，分析顺序：旧 → 新。")
m1,m2,m3,m4 = st.columns(4)
m1.metric("有效期数", f"{len(data):,}")
m2.metric("无效记录", f"{invalid:,}")
m3.metric("重复期号", f"{duplicates:,}")
m4.metric("平均和值", f"{data['和值'].mean():.2f}")
with st.expander("最近开奖记录与数据检查"):
    st.write(f"原始 {original:,} 条；有效 {len(data):,} 条。")
    st.dataframe(data.tail(30), use_container_width=True, hide_index=True)

# Descriptive matrices
st.subheader("1. 和值理论与历史频率矩阵")
window = st.select_slider("近期统计窗口", options=[20,50,100,200,300,500], value=100)
sums = data["和值"].to_numpy(dtype=int)
full = pd.Series(sums).value_counts().reindex(SUMS, fill_value=0)
recent = pd.Series(sums[-window:]).value_counts().reindex(SUMS, fill_value=0)
matrix = pd.DataFrame({
    "和值": SUMS, "理论概率(%)": THEORY*100,
    "全历史频率(%)": full.to_numpy()/len(sums)*100,
    f"近{min(window,len(sums))}期频率(%)": recent.to_numpy()/min(window,len(sums))*100,
})
st.dataframe(matrix.style.format({c:"{:.3f}" for c in matrix.columns if c!="和值"}),
             use_container_width=True, hide_index=True)
st.line_chart(matrix.set_index("和值"), use_container_width=True)

st.subheader("2. A / B / C 数字分布")
digits = pd.DataFrame({"数字": range(10)})
for col in ["A","B","C"]:
    ct = data[col].value_counts().reindex(range(10), fill_value=0)
    digits[col+"次数"] = ct.to_numpy()
    digits[col+"频率(%)"] = ct.to_numpy()/len(data)*100
st.dataframe(digits, use_container_width=True, hide_index=True)

st.subheader("3. 大小单双统计")
cats = pd.DataFrame({"类别":["大(14–27)","小(0–13)","单","双"],
                     "次数":[int((sums>=14).sum()),int((sums<=13).sum()),
                             int((sums%2==1).sum()),int((sums%2==0).sum())]})
cats["频率(%)"] = cats["次数"]/len(data)*100
st.dataframe(cats, use_container_width=True, hide_index=True)

st.subheader("4. V4.5 自适应参数验证与最终留出测试")
st.caption("严格按时间切分：前60%初始训练、接着20%只用于选参数、最后20%只用于最终测试。测试段不参与调参。")
result = adaptive_walk_forward(sums, data["期号"].astype(str).to_numpy())
if result is None:
    st.warning("有效数据不足。至少建议 500 期，且需要至少 30 期最终测试样本。")
else:
    train_end, val_end, tuning, summary, detail, comparison, best_w, best_s = result
    st.write(f"初始训练：{train_end:,} 期；参数验证：{val_end-train_end:,} 期；最终留出测试：{len(sums)-val_end:,} 期。")
    st.metric("验证段选出的参数", f"窗口 {best_w} 期 / 平滑强度 {best_s:g}")
    st.markdown("**最终留出测试结果**")
    st.dataframe(summary.style.format({"命中率(%)":"{:.2f}","平均和值差":"{:.3f}",
        "Log Loss（越低越好）":"{:.5f}","Brier Score（越低越好）":"{:.5f}"}),
        use_container_width=True, hide_index=True)
    st.markdown("**相对理论基准的逐期 Log Loss 差异（描述性参考）**")
    st.dataframe(comparison.style.format({"相对理论基准平均Log Loss差":"{:.5f}","差值标准误":"{:.5f}"}),
                 use_container_width=True, hide_index=True)
    with st.expander("查看所有候选参数在验证段的表现"):
        st.dataframe(tuning.head(20).style.format({"验证Log Loss":"{:.5f}","验证Brier":"{:.5f}"}),
                     use_container_width=True, hide_index=True)
    with st.expander("查看最终测试逐期明细"):
        st.dataframe(detail, use_container_width=True, hide_index=True)

st.subheader("5. 下一期理论和值概率参考")
st.caption("理论分布来自三个独立均匀数字相加的数学模型，不因历史冷热而改变。")
next_df = pd.DataFrame({"和值":SUMS, "理论概率(%)":THEORY*100})
next_df["理论排名"] = next_df["理论概率(%)"].rank(method="min", ascending=False).astype(int)
st.dataframe(next_df.sort_values(["理论排名","和值"]).style.format({"理论概率(%)":"{:.3f}"}),
             use_container_width=True, hide_index=True)

# Excel report
out = io.BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    data.to_excel(writer, sheet_name="历史数据", index=False)
    matrix.to_excel(writer, sheet_name="和值矩阵", index=False)
    digits.to_excel(writer, sheet_name="数字分布", index=False)
    cats.to_excel(writer, sheet_name="大小单双", index=False)
    if result is not None:
        summary.to_excel(writer, sheet_name="最终测试汇总", index=False)
        comparison.to_excel(writer, sheet_name="模型对照", index=False)
        tuning.to_excel(writer, sheet_name="验证参数排名", index=False)
        detail.to_excel(writer, sheet_name="逐期测试明细", index=False)
st.download_button("下载 V4.5 Excel 分析报告", out.getvalue(),
                   "Canada28_Matrix_AI_V4_5_Report.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
st.caption("数据来源：pc28.world 第三方公开 API，非 BCLC 官方网站。历史统计和回测不保证预测未来开奖结果，也不构成投注建议。")
