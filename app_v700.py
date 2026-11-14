# -*- coding: utf-8 -*-
"""Canada28 Matrix AI V7.0
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
DATA_DIR = Path("data_v700")
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
    page_title="Canada28 Matrix AI V7.0",
    page_icon="📊",
    layout="wide",
)
import hmac

def check_password():
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔐 Canada28 Matrix AI V7.0")
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
st.title("Canada28 Matrix AI V7.0｜全维度矩阵分析")
st.caption(
    "历史开奖数据驱动的三球与总和值统计研究。数据来自 pc28.world；"
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
    db_put("v700_history.csv", DATA_FILE.read_text(encoding="utf-8-sig"))


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



# V7: 14 base estimators + adaptive ensemble. All outputs are categorical distributions.
NAMES = ['理论基准','全历史贝叶斯','近500期','近200期','近100期','近50期','指数衰减','一阶马尔可夫','二阶马尔可夫','历史相似状态','随机森林','梯度提升','逻辑回归','三球联合矩阵']
BASE = np.bincount([a+b+c for a in range(10) for b in range(10) for c in range(10)], minlength=28)/1000
COMB = ['小双','小单','大双','大单']

def candidates(arr, col, half=500):
    arr=np.asarray(arr,dtype=int); y=arr[:,col] if col<3 else arr.sum(axis=1)
    base=np.full(10,.1) if col<3 else BASE; k=len(base)
    def smooth(v, strength=200, weights=None):
        counts=np.bincount(v,weights=weights,minlength=k)
        return (counts+strength*base)/(counts.sum()+strength)
    n=len(y); out=[base.copy(),smooth(y)]
    out += [smooth(y[-w:]) for w in (500,200,100,50)]
    out.append(smooth(y,100,np.power(.5,np.arange(n-1,-1,-1)/half)))
    for lag in (1,2,3):
        matches=[y[i] for i in range(lag,n) if np.array_equal(y[i-lag:i],y[-lag:])]
        out.append(smooth(np.asarray(matches,dtype=int),100))
    # Features contain only previous draws; no current target values.
    ml=[base.copy() for _ in range(3)]
    if n>=120:
        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        X=np.array([arr[i-3:i].ravel() for i in range(3,n)])[-750:]; Y=y[3:][-750:]
        if len(np.unique(Y))>1:
            models=[RandomForestClassifier(n_estimators=20,max_depth=4,min_samples_leaf=20,random_state=2026,n_jobs=1),HistGradientBoostingClassifier(max_iter=12,max_leaf_nodes=7,l2_regularization=10,random_state=2026),LogisticRegression(max_iter=200,C=.05)]
            for j,m in enumerate(models):
                m.fit(X,Y); p=np.zeros(k); p[m.classes_.astype(int)]=m.predict_proba(arr[-3:].reshape(1,-1))[0]; ml[j]=.5*p+.5*base
    out.extend(ml)
    # Empirical joint distribution of triples, shrunk to independent uniform triples.
    joint=np.bincount(arr[:,0]*100+arr[:,1]*10+arr[:,2],minlength=1000).astype(float)+.3
    ids=np.arange(1000); digits=np.array([ids//100,ids//10%10,ids%10]); labels=digits[col] if col<3 else digits.sum(axis=0)
    out.append(np.bincount(labels,weights=joint,minlength=k)/joint.sum())
    return np.array(out)

def groups(k, kind):
    x=np.arange(k); big=x >= (5 if k==10 else 14)
    if kind=='大小': return [x[~big],x[big]],['小','大']
    if kind=='单双': return [x[x%2==0],x[x%2==1]],['双','单']
    return [x[(big==b)&(x%2==o)] for b in (False,True) for o in (0,1)],COMB

def project(p,kind):
    if kind in ('六码','七码'):
        ix=np.argsort(-p,kind='stable')[:6 if kind=='六码' else 7]; return ''.join(map(str,ix)),float(p[ix].sum())
    g,l=groups(len(p),kind); q=np.array([p[i].sum() for i in g]); j=int(np.argmin(q) if kind=='杀组合' else np.argmax(q)); return l[j],float(1-q[j] if kind=='杀组合' else q[j])

def actual_label(x,k,kind):
    if kind in ('六码','七码'): return str(x)
    g,l=groups(k,kind); return next(label for ix,label in zip(g,l) if x in ix)

def hit(pick,actual,kind):
    return actual in pick if kind in ('六码','七码') else (pick!=actual if kind=='杀组合' else pick==actual)

SPECS=[(pos,kind) for pos in ('A','B','C','和值') for kind in (['大小','单双','组合','六码','七码'] if pos!='和值' else ['大小','单双','组合','杀组合'])]
KEY='v700_ledger.json'

def read_state():
    return json.loads(db_get(KEY) or '{}')

def transact(fn):
    # Serialize concurrent sessions for this ledger; never overwrite old version keys.
    conn,remote=db_connect(); ph='%s' if remote else '?'
    try:
        with conn:
            cur=conn.cursor()
            if not remote: cur.execute('BEGIN IMMEDIATE')
            cur.execute(f'INSERT INTO canada28_artifacts (artifact_key,content,updated_at) VALUES ({ph},{ph},{ph}) ON CONFLICT(artifact_key) DO NOTHING',(KEY,'{}',beijing_now().isoformat()))
            cur.execute(f'SELECT content FROM canada28_artifacts WHERE artifact_key={ph}'+(' FOR UPDATE' if remote else ''),(KEY,))
            state=json.loads(cur.fetchone()[0]); changed=fn(state)
            if changed: cur.execute(f'UPDATE canada28_artifacts SET content={ph},updated_at={ph} WHERE artifact_key={ph}',(json.dumps(state,ensure_ascii=False),beijing_now().isoformat(),KEY))
            return state
    finally: conn.close()

def settle(state,history):
    changed=False; draws=history.set_index('期号')
    for issue,rec in state.items():
        if int(issue) not in draws.index: continue
        d=draws.loc[int(issue)]
        for name,r in rec['projects'].items():
            pos,kind=name.split('|'); x=int(d[pos]); a=actual_label(x,28 if pos=='和值' else 10,kind)
            verdict='正确' if hit(r['pick'],a,kind) else '错误'
            if r.get('status')!=verdict or r.get('actual')!=a:
                r.update(actual=a,status=verdict,draw_time=format_beijing_time(d['开奖时间']),number=x); changed=True
    return changed

def weights_for(state,name,k,kind):
    rows=[r['projects'][name] for _,r in sorted(state.items(),key=lambda z:int(z[0])) if name in r['projects'] and r['projects'][name].get('status') in ('正确','错误')]
    n=len(rows); cutoff=n//50*50; rows=rows[:cutoff][-300:]; w=np.zeros(len(NAMES)); w[0]=1
    if not rows: return w,n
    losses=[]
    for r in rows:
        x=r['number']; probs=np.array(r['models']); a=actual_label(x,k,kind)
        losses.append([0 if hit(project(p,kind)[0],a,kind) else 1 for p in probs])
    loss=np.mean(losses,axis=0); raw=np.exp(-8*(loss-loss.min())); raw/=raw.sum()
    # Conservative baseline allocation; no evidence of superiority claimed.
    w=.5*raw; w[0]+=.5
    return w,n

def build_record(history,state,half):
    arr=history[['A','B','C']].to_numpy(dtype=int); models=[candidates(arr,i,half) for i in range(4)]
    projects={}; distributions={}
    for pos,kind in SPECS:
        i=('A','B','C','和值').index(pos); m=models[i]; name=pos+'|'+kind; w,n=weights_for(state,name,m.shape[1],kind); p=w@m
        pick,prob=project(p,kind)
        projects[name]={'pick':pick,'prob':prob,'p':p.tolist(),'models':m.tolist(),'weights':w.tolist(),'status':'等待开奖','calibrated':n//50*50}
    # Shared distribution for displayed categories: independent project weights are
    # retained for evaluation, but each project's probability is explicitly labeled.
    return {'time':beijing_now().isoformat(),'base':int(history['期号'].max()),'projects':projects}

def streaks(rows):
    runs=[]; active=None
    for r in sorted(rows,key=lambda z:z['期号']):
        issue=r['期号']; status=r['状态']
        if active and (issue!=active['结束期号']+1 or status!=active['结果']):
            active['状态']='已结束' if status in ('正确','错误') and issue==active['结束期号']+1 else '中断/待核对'; runs.append(active); active=None
        if status not in ('正确','错误'): continue
        if active is None: active={'结果':status,'开始期号':issue,'结束期号':issue,'长度':1,'开始时间':r.get('开奖时间',''),'结束时间':r.get('开奖时间',''),'状态':'当前'}
        else: active.update(结束期号=issue,长度=active['长度']+1,结束时间=r.get('开奖时间',''))
    if active: runs.append(active)
    return runs

def legacy_rows(name):
    pos,kind=name.split('|'); filename={'六码':'v64_six_digit_log.csv','七码':'v61_seven_digit_log.csv'}.get(kind)
    if not filename: return []
    raw=db_get(filename)
    if not raw: return []
    f=pd.read_csv(io.StringIO(raw),dtype=str).fillna(''); result=[]
    for _,r in f.iterrows():
        try:
            issue=int(r['目标期号']); pick=r[pos+kind]; a=r[pos+'实际号码']; status=r[pos+'是否命中']
            if int(r['基于最新期号'])+1!=issue or len(pick)!=int(6 if kind=='六码' else 7) or not pick.isdigit() or len(set(pick))!=len(pick): continue
            if a in list('0123456789'): status='正确' if a in pick else '错误'
            else: status='等待开奖'
            result.append({'期号':issue,'预测时间':format_beijing_time(r['预测时间']),'开奖时间':'','预测':pick,'实际':a,'状态':status,'来源':'旧版'})
        except (ValueError,KeyError): continue
    return result

# UI and sync. No writes to legacy prediction artifacts.
with st.sidebar:
    st.header('系统设置'); half=st.slider('半衰期',100,2000,500,50); auto=st.checkbox('自动同步',True); count=st.selectbox('首次同步',[1000,3000,5000],index=2)
try:
    remote=db_init()
    if not remote: st.error('V7.0 要求配置 DATABASE_URL，避免历史记录丢失。'); st.stop()
except Exception as e:
    st.error('数据库连接失败：'+type(e).__name__); st.stop()
if 'v700_history' not in st.session_state:
    raw=db_get('v700_history.csv') or db_get('canada28_history.csv')
    st.session_state.v700_history=pd.read_csv(io.StringIO(raw)) if raw else pd.DataFrame()
    st.session_state.v700_sync=0

@st.fragment(run_every=5 if auto else None)
def poll():
    h=st.session_state.v700_history
    try:
        latest=fetch_latest(); now=time.time(); full=now-st.session_state.v700_sync>=210
        if not latest: raise ValueError('最新开奖为空')
        if h.empty or full or int(latest['期号'])>int(h['期号'].max())+1:
            incoming=fetch_history(count if h.empty else 200); st.session_state.v700_sync=now
        else: incoming=pd.DataFrame([latest])
        new=merge_data(h,merge_data(incoming,pd.DataFrame([latest])))
        if not new.equals(h):
            st.session_state.v700_history=new; save_local(new)
            transact(lambda state:settle(state,new)); st.rerun(scope='app')
        st.caption('最新开奖检查每5秒；历史补同步每210秒。')
    except Exception as e: st.warning('同步暂时失败：'+type(e).__name__)
if auto or st.button('手动同步'): poll()
h=st.session_state.v700_history
if h.empty: st.stop()
h=h.sort_values('期号').drop_duplicates('期号',keep='last'); latest=int(h.iloc[-1]['期号']); target=str(latest+1)
d=h.iloc[-1]; st.subheader(f"最新第{latest}期：{d['A']}+{d['B']}+{d['C']}={d['和值']}"); st.caption(format_beijing_time(d['开奖时间']))
@st.fragment(run_every=5)
def countdown():
    t=parse_beijing_time(d['开奖时间']); remaining=max(0,int((t+timedelta(seconds=210)-beijing_now()).total_seconds())) if t else None
    st.write(f'预计下一期开奖倒计时：{remaining}秒' if remaining is not None else '开奖时间未知')
countdown()
state=transact(lambda s:settle(s,h))
# Confirm latest source again before locking; do not backfill if source is unavailable.
if target not in state:
    try:
        with st.spinner('计算14种候选模型及动态融合…'):
            record=build_record(h,state,half)
            fresh=fetch_latest(); deadline=parse_beijing_time(d['开奖时间'])
            if not fresh or int(fresh['期号'])!=latest or not deadline or beijing_now()>=deadline+timedelta(seconds=210):
                st.warning('最新期号或时间无法确认，暂停锁定，等待同步。')
            else:
                def lock(s):
                    if target in s: return False
                    s[target]=record; return True
                state=transact(lock)
    except Exception as e: st.error('模型计算或锁定失败：'+type(e).__name__)
st.header(f'下一期 {target}｜事前锁定预测')
if target in state:
    rec=state[target]; rows=[]
    for name,r in rec['projects'].items():
        rows.append({'项目':name,'方向':r['pick'],'估计概率':f"{r['prob']:.2%}",'已校准':r['calibrated']})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    name=st.selectbox('查看概率与模型权重',list(rec['projects']))
    r=rec['projects'][name]; st.bar_chart(pd.DataFrame({'概率':r['p']})); st.dataframe(pd.DataFrame({'模型':NAMES,'权重':r['weights']}),hide_index=True)
    kind=name.split('|')[1]
    if kind not in ('六码','七码'):
        g,l=groups(len(r['p']),kind); st.dataframe(pd.DataFrame({'类别':l,'概率':[sum(r['p'][i] for i in ix) for ix in g]}),hide_index=True)
st.header('全项目历史成绩与连续错误登记')
allrows={}; summary=[]; runrows=[]
for pos,kind in SPECS:
    name=pos+'|'+kind; rows={r['期号']:r for r in legacy_rows(name)}
    for issue,rec in state.items():
        r=rec['projects'].get(name)
        if r and int(issue) not in rows: rows[int(issue)]={'期号':int(issue),'预测时间':format_beijing_time(rec['time']),'开奖时间':r.get('draw_time',''),'预测':r['pick'],'实际':r.get('actual',''),'状态':r['status'],'来源':'V7.0'}
    rows=sorted(rows.values(),key=lambda r:r['期号']); allrows[name]=rows; runs=streaks(rows)
    checked=[r for r in rows if r['状态'] in ('正确','错误')]; row={'项目':name,'已核对':len(checked),'命中率':sum(r['状态']=='正确' for r in checked)/len(checked) if checked else None}
    for status in ('正确','错误'):
        rr=[r for r in runs if r['结果']==status]; m=max([r['长度'] for r in rr],default=0); row['最高'+status]=m; row['最高'+status+'次数']=sum(r['长度']==m for r in rr)
    summary.append(row); runrows.extend([dict(r,项目=name) for r in runs])
st.dataframe(pd.DataFrame(summary),hide_index=True)
name=st.selectbox('历史项目',list(allrows)); rows=allrows[name]; runs=pd.DataFrame([r for r in runrows if r['项目']==name]); st.dataframe(runs,hide_index=True)
if not runs.empty: st.dataframe(runs.groupby(['结果','长度','状态']).size().reset_index(name='出现次数'),hide_index=True)
st.download_button('下载全部连中连错登记',pd.DataFrame(runrows).to_csv(index=False).encode('utf-8-sig'),'v700_streaks.csv','text/csv')
if rows:
    frame=pd.DataFrame(rows); buckets=sorted({r['期号']//100*100 for r in rows},reverse=True); bucket=st.selectbox('连续100期档案',buckets,format_func=lambda x:f'{x}–{x+99}')
    archive=frame.set_index('期号').reindex(range(bucket,bucket+100)).rename_axis('期号').reset_index(); archive['状态']=archive['状态'].fillna('未记录预测'); st.dataframe(archive,hide_index=True)
    st.download_button('下载项目完整历史',frame.to_csv(index=False).encode('utf-8-sig'),'v700_history_project.csv','text/csv')
st.caption('历史成绩不代表未来命中率。网页运行期间才会自动同步；旧版记录只读。')
