# -*- coding: utf-8 -*-
"""Canada28 Matrix AI V7.8.1
七模块独立分析工具。仅作统计研究，不保证预测准确。
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
DATA_DIR = Path("data_v710")
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
    page_title="Canada28 Matrix AI V7.8.1",
    page_icon="📊",
    layout="wide",
)
import hmac

def check_password():
    if st.session_state.get("authenticated", False):
        return True

    st.title("🔐 Canada28 Matrix AI V7.8.1")
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
st.title("Canada28 Matrix AI V7.8.1｜661经典布局全功能版")
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
    db_put("v710_history.csv", DATA_FILE.read_text(encoding="utf-8-sig"))


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

def candidates(arr, col, half=500, fitted=None):
    arr=np.asarray(arr,dtype=int); y=arr[:,col] if col<3 else arr.sum(axis=1)
    base=np.full(10,.1) if col<3 else BASE; k=len(base)
    def smooth(v, strength=200, weights=None):
        counts=np.bincount(v,weights=weights,minlength=k)
        return (counts+strength*base)/(counts.sum()+strength)
    n=len(y); out=[base.copy(),smooth(y)]
    out += [smooth(y[-w:]) for w in (500,200,100,50)]
    out.append(smooth(y,100,np.power(.5,np.arange(n-1,-1,-1)/half)))
    for lag in (1,2,3):
        matches=y[lag:][np.all(np.lib.stride_tricks.sliding_window_view(y,lag)[:-1] == y[-lag:],axis=1)] if n>lag else np.array([],dtype=int)
        out.append(smooth(np.asarray(matches,dtype=int),100))
    # Features contain only previous draws; no current target values.
    ml=[base.copy() for _ in range(3)]
    if n>=120:
        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        X=np.array([arr[i-3:i].ravel() for i in range(3,n)])[-750:]; Y=y[3:][-750:]
        if len(np.unique(Y))>1:
            models=[RandomForestClassifier(n_estimators=20,max_depth=4,min_samples_leaf=20,random_state=2026,n_jobs=1),HistGradientBoostingClassifier(max_iter=12,max_leaf_nodes=7,l2_regularization=10,random_state=2026),LogisticRegression(max_iter=200,C=.05)]
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=1):
                if fitted is not None and col in fitted:
                    models=fitted[col]
                else:
                    for m in models: m.fit(X,Y)
                    if fitted is not None: fitted[col]=models
                for j,m in enumerate(models):
                    p=np.zeros(k); p[m.classes_.astype(int)]=m.predict_proba(arr[-3:].reshape(1,-1))[0]; ml[j]=.5*p+.5*base
    out.extend(ml)
    # Empirical joint distribution of triples, shrunk to independent uniform triples.
    joint_ids=arr[:,0]*100+arr[:,1]*10+arr[:,2]
    bands=arr.sum(axis=1)//7
    matched=joint_ids[1:][bands[:-1]==bands[-1]] if n>1 else np.array([],dtype=int)
    joint=np.bincount(matched,minlength=1000).astype(float)+.3
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

MODULES={
    'V7.1 三球大小': [(p,'大小') for p in ('A','B','C')],
    'V7.2 三球单双': [(p,'单双') for p in ('A','B','C')],
    'V7.3 三球组合': [(p,'组合') for p in ('A','B','C')],
    'V7.4 总和值大小': [('和值','大小')],
    'V7.5 总和值单双': [('和值','单双')],
    'V7.6 总和值组合': [('和值','组合')],
    'V7.7 总和值杀组合': [('和值','杀组合')],
}
SPECS=[spec for specs in MODULES.values() for spec in specs]
ACTIVE={pos+'|'+kind for pos,kind in SPECS}
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
            if name not in ACTIVE: continue
            pos,kind=name.split('|'); x=int(d[pos]); a=actual_label(x,28 if pos=='和值' else 10,kind)
            verdict='正确' if hit(r['pick'],a,kind) else '错误'
            if r.get('status')!=verdict or r.get('actual')!=a:
                r.update(actual=a,status=verdict,draw_time=format_beijing_time(d['开奖时间']),number=x); changed=True
    return changed

INIT_KEY='v780_initial_validation.json'
KILL='和值|杀组合'
KILL_NAMES=NAMES+['固定杀大单','固定杀小双','等权历史融合']

def kill_losses(models,x):
    a=actual_label(x,28,'杀组合')
    picks=[project(p,'杀组合')[0] for p in models]
    picks += ['大单','小双',project(.5*models[0]+.5*models[1:].mean(axis=0),'杀组合')[0]]
    return [int(p==a) for p in picks]

def kill_weights(losses):
    # Select on earlier observations, confirm once on later observations.
    w=np.zeros(len(NAMES)); w[0]=1
    a=np.asarray(losses,dtype=float)
    if len(a)<40: return w,'固定杀小双（理论基准）'
    mid=len(a)//2; train=a[:mid].mean(axis=0); confirm=a[mid:].mean(axis=0)
    j=min(list(range(1,14))+[16],key=lambda j:train[j])
    # Require at least 2 percentage points better than BOTH fixed baselines in both halves.
    if train[j]+.02<=min(train[14:16]) and confirm[j]+.02<=min(confirm[14:16]):
        w[:]=0
        if j==16: w[0]=.5; w[1:]=.5/13
        else: w[j]=1
        return w,KILL_NAMES[j]
    return w,'固定杀小双（理论基准）'


def validation_loss(models,x,kind):
    # Proper event Brier for categories; coverage decisions use hit loss.
    k=models.shape[1]; a=actual_label(x,k,kind)
    if kind in ('六码','七码','杀组合'):
        return np.array([not hit(project(p,kind)[0],a,kind) for p in models],dtype=float)
    g,l=groups(k,kind); q=np.array([[p[ix].sum() for ix in g] for p in models]); y=np.array([v==a for v in l],dtype=float)
    return np.sum((q-y)**2,axis=1)

def initial_validation(history,half):
    arr=history[['A','B','C']].to_numpy(dtype=int); issues=history['期号'].to_numpy(dtype=int)
    # 120 chronological targets, sampled every 3 draws; train only on earlier data.
    indices=[i for i in range(max(120,len(arr)-360),len(arr),3) if issues[i]==issues[i-1]+1]
    if len(indices)<40: return None
    scores={pos+'|'+kind:[] for pos,kind in SPECS}; fitted={}; kill_scores=[]
    for j,i in enumerate(indices):
        if j%20==0: fitted={}
        m=[candidates(arr[:i],c,half,fitted) for c in range(4)]
        kill_scores.append(kill_losses(m[3],int(arr[i].sum())))
        for pos,kind in SPECS:
            c=('A','B','C','和值').index(pos); x=int(arr[i,c] if c<3 else arr[i].sum())
            scores[pos+'|'+kind].append(validation_loss(m[c],x,kind).tolist())
    return {'version':1,'half':half,'last':int(issues[-1]),'history_n':len(arr),'n':len(indices),'scores':scores,'kill_scores':kill_scores}

def initial_state(history,half):
    saved=json.loads(db_get(INIT_KEY) or '{}')
    if saved.get('version')==1 and saved.get('half')==half and saved.get('n',0)>=40: return saved
    result=initial_validation(history,half)
    if result: db_put(INIT_KEY,json.dumps(result,ensure_ascii=False))
    return result or {}

def choose_weights(losses):
    w=np.zeros(len(NAMES)); w[0]=1
    if len(losses)<40: return w
    a=np.asarray(losses); mid=len(a)//2; train=a[:mid].mean(axis=0); confirm=a[mid:].mean(axis=0)
    # Select on first half; confirm on second half. Conservative margin, not proof.
    eligible=[j for j in range(1,len(NAMES)) if train[j]+.005<train[0] and confirm[j]+.005<confirm[0]]
    if eligible:
        raw=np.exp(-8*(train[eligible]-min(train[eligible]))); raw/=raw.sum(); w[0]=.5; w[eligible]=.5*raw
    return w

def weights_for(state,name,k,kind,initial):
    rows=[r['projects'][name] for _,r in sorted(state.items(),key=lambda z:int(z[0])) if name in r['projects'] and r['projects'][name].get('status') in ('正确','错误')]
    n=len(rows); cutoff=n//50*50; rows=rows[:cutoff][-300:]
    losses=[validation_loss(np.array(r['models']),r['number'],kind) for r in rows]
    if len(losses)<50: losses=initial.get('scores',{}).get(name,[])
    if name==KILL:
        kl=[kill_losses(np.asarray(r['models']),r['number']) for r in rows] if len(rows)>=50 else initial.get('kill_scores',[])
        return kill_weights(kl)[0],n
    return choose_weights(losses),n

def build_record(history,state,half,initial):
    arr=history[['A','B','C']].to_numpy(dtype=int); models=[candidates(arr,i,half) for i in range(4)]
    projects={}
    for pos,kind in SPECS:
        i=('A','B','C','和值').index(pos); m=models[i]; name=pos+'|'+kind; w,n=weights_for(state,name,m.shape[1],kind,initial); p=w@m
        pick,prob=project(p,kind)
        projects[name]={'pick':pick,'prob':prob,'p':p.tolist(),'models':m.tolist(),'weights':w.tolist(),'status':'等待开奖','calibrated':n//50*50}
    return {'time':beijing_now().isoformat(),'base':int(history['期号'].max()),'projects':projects,'version':'V7.8.1','initial_n':initial.get('n',0)}

def streaks(rows):
    runs=[]; active=None
    ordered=sorted(rows,key=lambda z:z['期号'])
    while ordered and ordered[-1]['状态']=='等待开奖': ordered.pop()
    for r in ordered:
        issue=r['期号']; status=r['状态']
        if active and (issue!=active['结束期号']+1 or status!=active['结果']):
            active['状态']='已结束' if status in ('正确','错误') and issue==active['结束期号']+1 else '中断/待核对'; runs.append(active); active=None
        if status not in ('正确','错误'): continue
        if active is None: active={'结果':status,'开始期号':issue,'结束期号':issue,'长度':1,'开始时间':r.get('开奖时间',''),'结束时间':r.get('开奖时间',''),'状态':'当前'}
        else: active.update(结束期号=issue,长度=active['长度']+1,结束时间=r.get('开奖时间',''))
    if active: runs.append(active)
    return runs

def display_status(frame):
    # Presentation only: preserve canonical database and CSV status values.
    frame=frame.copy()
    symbols={'正确':'✅️','错误':'❌️','等待开奖':'⏳ 等待开奖'}
    for col in ('状态','结果'):
        if col in frame.columns: frame[col]=frame[col].replace(symbols)
    return frame

# UI and sync. No writes to legacy prediction artifacts.
with st.sidebar:
    st.header('综合仪表盘'); module='全部13项'; visible=SPECS; st.caption('全部13项同页显示、同步锁定。'); st.header('系统设置'); half=st.slider('半衰期',100,2000,500,50); auto=st.checkbox('自动同步',True); count=st.selectbox('首次同步',[1000,3000,5000],index=2)
try:
    remote=db_init()
    if not remote: st.error('V7.8.1 要求配置 DATABASE_URL，避免历史记录丢失。'); st.stop()
except Exception as e:
    st.error('数据库连接失败：'+type(e).__name__); st.stop()
if 'v700_history' not in st.session_state:
    raw=db_get('v710_history.csv') or db_get('v700_history.csv') or db_get('canada28_history.csv')
    st.session_state.v700_history=pd.read_csv(io.StringIO(raw)) if raw else pd.DataFrame()
    st.session_state.v700_sync=0

@st.fragment(run_every=5 if auto else None)
def poll():
    h=st.session_state.v700_history
    try:
        latest=fetch_latest(); now=time.time(); full=now-st.session_state.v700_sync>=210
        if not latest: raise ValueError('最新开奖为空')
        if h.empty or len(h)<count or full or int(latest['期号'])>int(h['期号'].max())+1:
            incoming=fetch_history(count if h.empty or len(h)<count else 200); st.session_state.v700_sync=now
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
initial={}
try:
    with st.spinner('首次历史回测：14种模型，严格时间顺序…'):
        initial=initial_state(h,half)
except Exception as e:
    st.warning('历史初始化失败，暂用理论基准：'+type(e).__name__)
st.caption(f"历史开奖：{len(h)}期；初始回测：{initial.get('n',0)}个抽样目标；训练截止：{initial.get('last','未完成')}")
if initial:
    with st.expander('初始历史回测结果（非真实预测成绩）'):
        selected=st.selectbox('回测项目',[p+'|'+k for p,k in visible],key='v770_validation_project_'+module)
        a=np.array(initial['scores'][selected]); mid=len(a)//2
        st.dataframe(pd.DataFrame({'模型':NAMES,'选择段损失':a[:mid].mean(axis=0),'确认段损失':a[mid:].mean(axis=0),'初始权重':kill_weights(initial.get('kill_scores',[]))[0] if selected==KILL else choose_weights(a)}),hide_index=True)
# Confirm latest source again before locking; do not backfill if source is unavailable.
if target not in state:
    try:
        with st.spinner('计算14种候选模型及动态融合…'):
            record=build_record(h,state,half,initial)
            fresh=fetch_latest(); deadline=parse_beijing_time(d['开奖时间'])
            if not fresh or int(fresh['期号'])!=latest or not deadline or beijing_now()>=deadline+timedelta(seconds=210):
                st.warning('最新期号或时间无法确认，暂停锁定，等待同步。')
            else:
                def lock(s):
                    if target in s or beijing_now()>=deadline+timedelta(seconds=210): return False
                    record['time']=beijing_now().isoformat(); s[target]=record; return True
                state=transact(lock)
    except Exception as e: st.error('模型计算或锁定失败：'+type(e).__name__)
allrows={}; summary=[]; runrows=[]
for pos,kind in visible:
    name=pos+'|'+kind; rows={}
    for issue,rec in state.items():
        r=rec['projects'].get(name)
        if r and int(issue) not in rows: rows[int(issue)]={'期号':int(issue),'预测时间':format_beijing_time(rec['time']),'开奖时间':r.get('draw_time',''),'预测':r['pick'],'实际':r.get('actual',''),'状态':r['status'],'来源':rec.get('version','V7.0'),'锁定概率':r.get('prob'),'模型权重':json.dumps(dict(zip(NAMES,r.get('weights',[]))),ensure_ascii=False)}
    rows=sorted(rows.values(),key=lambda r:r['期号']); allrows[name]=rows; runs=streaks(rows)
    checked=[r for r in rows if r['状态'] in ('正确','错误')]; row={'项目':name,'已核对':len(checked),'命中率':sum(r['状态']=='正确' for r in checked)/len(checked) if checked else None}
    for status in ('正确','错误'):
        rr=[r for r in runs if r['结果']==status]; m=max([r['长度'] for r in rr],default=0); row['最高'+status]=m; row['最高'+status+'次数']=sum(r['长度']==m for r in rr)
    for window in (10,20,50,100,200,500):
        sample=checked[-window:]; row[f'近{window}期命中率']=sum(r['状态']=='正确' for r in sample)/len(sample) if sample else None; row[f'近{window}期样本']=len(sample)
    current=next((r for r in reversed(runs) if r['状态']=='当前'),None); row['当前连错']=current['长度'] if current and current['结果']=='错误' else 0; row['当前连中']=current['长度'] if current and current['结果']=='正确' else 0
    summary.append(row); runrows.extend([dict(r,项目=name) for r in runs])
st.header(f'下一期 {target}｜事前锁定预测')
rec=state.get(target)
if rec:
    st.caption('锁定版本：'+rec.get('version','V7.0')+'；旧预测不重新计算。')
else:
    st.info('本期尚未锁定，请等待最新开奖确认。')
st.caption('持久化存储：PostgreSQL 已连接｜所有13项同步锁定；以下分区仅用于展示。')

def pct(v):
    return '—' if v is None or pd.isna(v) else f'{v:.2%}'

def show_module(label,specs):
    st.divider()
    st.subheader(label)
    names=[pos+'|'+kind for pos,kind in specs]
    cols=st.columns(len(specs))
    for col,(pos,kind),name in zip(cols,specs,names):
        with col:
            st.markdown('#### '+({'A':'第1球 A','B':'第2球 B','C':'第3球 C','和值':'总和值'}[pos]))
            r=rec.get('projects',{}).get(name) if rec else None
            if not r:
                st.info('本期未记录预测')
                continue
            st.metric('锁定排除组合' if kind=='杀组合' else '锁定方向',r['pick'],pct(r['prob']),delta_color='off')
            g,labels=groups(len(r['p']),kind); p=np.asarray(r['p']); probs=[float(p[ix].sum()) for ix in g]
            st.caption(' ｜ '.join(f'{a} {b:.2%}' for a,b in zip(labels,probs)))
            if sum(abs(q-(min(probs) if kind=='杀组合' else max(probs)))<1e-12 for q in probs)>1:
                st.caption('概率并列：固定顺序选择，无额外优势。')
            with st.expander('锁定模型权重与数字概率'):
                st.dataframe(pd.DataFrame({'模型':NAMES,'权重':r['weights']}),hide_index=True)
                st.bar_chart(pd.DataFrame({'概率':r['p']}))
    if specs[0][1]=='杀组合':
        st.caption('理论最佳固定排除基准76.9%；实际组合不等于排除组合即正确。')
    ss=[r for r in summary if r['项目'] in names]
    st.markdown('**实际成绩与连中连错**')
    scores=[]; calibration=[]
    for row in ss:
        name=row['项目']; n=row['已核对']; checked=[r for r in allrows[name] if r['状态'] in ('正确','错误')]; hits=sum(r['状态']=='正确' for r in checked)
        scores.append({'项目':name,'已核对':n,'正确':hits,'错误':n-hits,'命中率':pct(row['命中率']),'当前连中':row['当前连中'],'当前连错':row['当前连错'],'最高连中':row['最高正确'],'最高连错':row['最高错误'],'最高连错次数':row['最高错误次数']})
        calibration.append({'项目':name,'已核对':n,'累计满50期批次':n//50,'距离下一批':50-n%50})
    st.dataframe(pd.DataFrame(scores),hide_index=True)
    with st.expander('独立校准状态'):
        st.dataframe(pd.DataFrame(calibration),hide_index=True)
        st.caption('批次按已核对预测计数；锁定时实际采用的权重见各项目模型明细。')
    st.markdown('**最近预测记录与开奖核对**')
    recent=sorted({r['期号'] for name in names for r in allrows[name]},reverse=True)[:20]
    maps={name:{r['期号']:r for r in allrows[name]} for name in names}
    def combined(issues):
        result=[]
        for issue in issues:
            row={'期号':issue}
            for name in names:
                prefix=name.split('|')[0]; r=maps[name].get(issue,{})
                row[prefix+'预测']=r.get('预测',''); row[prefix+'实际']=r.get('实际',''); row[prefix+'核对']={'正确':'✅️','错误':'❌️','等待开奖':'⏳ 等待开奖'}.get(r.get('状态'),'未记录预测')
            result.append(row)
        return pd.DataFrame(result)
    if recent: st.dataframe(combined(recent),hide_index=True)
    else: st.info('暂无事前预测记录。')
    with st.expander('最近10/20/50/100/200/500期走势'):
        st.dataframe(pd.DataFrame([{'项目':r['项目'],'窗口':w,'样本':r[f'近{w}期样本'],'命中率':pct(r[f'近{w}期命中率'])} for r in ss for w in (10,20,50,100,200,500)]),hide_index=True)
    with st.expander('全部连中连错登记与起止期号'):
        runs=pd.DataFrame([r for r in runrows if r['项目'] in names])
        if not runs.empty:
            st.dataframe(display_status(runs.groupby(['项目','结果','长度']).size().reset_index(name='出现次数')),hide_index=True)
            st.dataframe(display_status(runs),hide_index=True)
            st.download_button('下载连中连错CSV',runs.to_csv(index=False).encode('utf-8-sig'),'v781_'+label.split()[0]+'_streaks.csv','text/csv',key='runs_'+label)
        else: st.info('暂无已核对记录。')
    with st.expander('连续100期档案与完整历史下载'):
        issues=[r['期号'] for name in names for r in allrows[name]]
        if issues:
            buckets=list(range(max(max(issues),latest)//100*100,min(issues)//100*100-1,-100))
            bucket=st.selectbox('选择100期档案',buckets,format_func=lambda x:f'{x}–{x+99}',key='bucket_'+label)
            archive=combined(range(bucket,bucket+100)); st.dataframe(archive,hide_index=True,height=440)
            st.download_button('下载100期档案',archive.to_csv(index=False).encode('utf-8-sig'),f'v781_{bucket}_{label.split()[0]}.csv','text/csv',key='archive_'+label)
            full=pd.DataFrame([dict(r,项目=name) for name in names for r in allrows[name]])
            st.download_button('下载完整历史（含锁定概率和权重）',full.to_csv(index=False).encode('utf-8-sig'),'v781_'+label.split()[0]+'_history.csv','text/csv',key='history_'+label)
        else: st.info('暂无历史档案。')
    if specs[0][1]=='杀组合':
        with st.expander('杀组合专项：真实成绩与固定基准对照',expanded=True):
            checked=[r for r in allrows[KILL] if r['状态'] in ('正确','错误')]
            comparisons=[]
            for window in (10,20,50,100,200,500,len(checked)):
                sample=checked[-window:] if window else []
                if not sample: continue
                n=len(sample)
                comparisons.append({'范围': '全部' if window==len(checked) else f'近{window}期','样本':n,'实际成功率':sum(r['状态']=='正确' for r in sample)/n,'固定杀大单对照':sum(r['实际']!='大单' for r in sample)/n,'固定杀小双对照':sum(r['实际']!='小双' for r in sample)/n})
            st.dataframe(pd.DataFrame(comparisons),hide_index=True)
            st.caption('固定基准为同一批实际开奖的事后对照，不是已锁定预测。理论基准76.9%，不保证未来成绩。')
            kl=np.asarray(initial.get('kill_scores',[]))
            if len(kl):
                mid=len(kl)//2; w,strategy=kill_weights(kl)
                st.write('初始专项回测策略：'+strategy)
                st.dataframe(pd.DataFrame({'策略':KILL_NAMES,'选择段成功率':1-kl[:mid].mean(axis=0),'确认段成功率':1-kl[mid:].mean(axis=0)}),hide_index=True)
            live=[r['projects'][KILL] for _,r in sorted(state.items(),key=lambda z:int(z[0])) if KILL in r['projects'] and r['projects'][KILL].get('status') in ('正确','错误') and 'models' in r['projects'][KILL] and 'number' in r['projects'][KILL]]
            if live:
                a=np.asarray([kill_losses(np.asarray(r['models']),r['number']) for r in live[-500:]])
                st.caption(f'候选模型事前保存概率的对照成绩：{len(a)}期；不是实际锁定成绩。')
                st.dataframe(pd.DataFrame({'策略':KILL_NAMES,'对照成功率':1-a.mean(axis=0)}),hide_index=True)

for label,specs in MODULES.items():
    show_module(label,specs)
st.divider()
st.caption('✅️ 正确　❌️ 错误　⏳ 等待开奖。历史成绩不代表未来命中率。网页运行期间才会自动同步；旧版记录只读。')
