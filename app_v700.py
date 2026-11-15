import os, json, math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title='Canada28 Matrix AI V7.0',layout='wide')
st.title('Canada28 Matrix AI V7.0')
st.caption('统计研究：随机开奖不能可靠预测；候选覆盖率不等于盈利概率。')
TZ=timezone(timedelta(hours=8)); BASE='https://pc28.world/api/v1'
URL=os.getenv('DATABASE_URL','')
try: URL=st.secrets.get('DATABASE_URL',URL)
except Exception: pass
@st.cache_resource
def db(url):
    e=create_engine(url.replace('postgres://','postgresql+psycopg://',1).replace('postgresql://','postgresql+psycopg://',1) if url else 'sqlite:///v700_local.db')
    with e.begin() as c:
        c.execute(text('CREATE TABLE IF NOT EXISTS v700_draws (issue BIGINT PRIMARY KEY, stamp TEXT, a INTEGER, b INTEGER, c INTEGER)'))
        c.execute(text('CREATE TABLE IF NOT EXISTS v700_predictions (issue BIGINT PRIMARY KEY, created TEXT, payload TEXT, actual TEXT)'))
    return e
E=db(URL)

def parse(raw):
    if isinstance(raw,dict):
        for k in ('data','draws','results','items','records','list'):
            if k in raw:return parse(raw[k])
        raw=[raw]
    out=[]
    for r in raw:
        try:
            def get(*keys):return next(r[k] for k in keys if k in r)
            i=int(get('issue','draw_id','period','nbr','期号','id','expect'))
            try: nums=[int(get('A','a','num1','n1','number1','号码1')),int(get('B','b','num2','n2','number2','号码2')),int(get('C','c','num3','n3','number3','号码3'))]
            except Exception:
                v=get('numbers','nums','result','open_code','opencode'); nums=list(map(int,v.replace('+',',').split(',') if isinstance(v,str) else v))
            if len(nums)!=3 or not all(0<=x<=9 for x in nums):continue
            try:s=str(get('draw_time','time','开奖时间','timestamp','open_time','created_at'))
            except Exception:s=''
            out.append(dict(issue=i,stamp=s,a=nums[0],b=nums[1],c=nums[2]))
        except Exception:continue
    return out

def save(rows):
    with E.begin() as c:
        for r in rows:c.execute(text('INSERT INTO v700_draws(issue,stamp,a,b,c) VALUES(:issue,:stamp,:a,:b,:c) ON CONFLICT(issue) DO NOTHING'),r)

def fetch(path,params=None):
    r=requests.get(BASE+path,params=params,timeout=15);r.raise_for_status();return parse(r.json())

def sync(n):
    seen=set()
    for page in range(1, min(100,math.ceil(n/100)+2)+1):
        rows=fetch('/draws',{'limit':100,'page':page}); new=[r for r in rows if r['issue'] not in seen]
        if not new:break
        save(new);seen.update(r['issue'] for r in new)
        if len(seen)>=n:break
    return len(seen)

def history():
    with E.connect() as c:return pd.read_sql(text('SELECT * FROM v700_draws ORDER BY issue DESC LIMIT 5000'),c).sort_values('issue').reset_index(drop=True)

# Each feature is computed only from draws available before the target.
NAMES=['均匀基准','加法','减法','乘法','除法','横向','纵向','正斜向','反斜向','反纵向','近期频率','贝叶斯','代数','几何','离散微积分','状态转移']
def features(x):
    u,v,w=x[-1],x[-2],x[-3]; j=np.arange(3); f=np.stack([u,(u+v)%10,(u-v)%10,(u*v)%10,np.floor(u/(v+1)).astype(int)%10,np.roll(u,1),v,(u+np.roll(v,1)+np.roll(w,2))%10,(u+np.roll(v,-1)+np.roll(w,-2))%10,(w-v)%10,u,u,(2*u-v)%10,np.floor(np.sqrt((u-v)**2+(v-w)**2)).astype(int)%10,(u-2*v+w)%10,u]);return f

def models(x, counts):
    p=(counts+3)/(counts.sum(axis=-1,keepdims=True)+30)
    f=features(x); out=np.stack([p[m,np.arange(3),f[m]] for m in range(len(NAMES))]);out[0]=.1
    for m,win in [(10,100),(11,500)]:
        z=x[-win:];out[m]=np.stack([(np.bincount(z[:,j],minlength=10)+20)/(len(z)+200) for j in range(3)])
    return out

def train(x):
    M=len(NAMES); counts=np.zeros((M,3,10,10)); loss=np.zeros((M,3)); weights=np.ones((M,3))/M; records=[]
    for t in range(3,len(x)):
        f=features(x[:t]); y=x[t]; p=models(x[:t],counts)
        mix=(p*weights[:,:,None]).sum(axis=0)
        if t>=100:
            order=np.argsort(-mix,axis=1); hit7=(order[:,:7]==y[:,None]).any(1);hit8=(order[:,:8]==y[:,None]).any(1)
            joint=np.einsum('i,j,k->ijk',*mix); big=joint[SUM>=14].sum(); odd=joint[SUM%2==1].sum(); records.append([*hit7,*hit8,hit7.all(),hit8.all(),(big>=.5)==(y.sum()>=14),(odd>=.5)==(y.sum()%2==1)])
        loss=.995*loss-.005*np.log(np.maximum(p[:,np.arange(3),y],1e-12))
        for m in range(M):counts[m,np.arange(3),f[m],y]+=1
        if (t+1)%50==0:
            z=np.exp(-8*(loss-loss.min(axis=0)));weights=z/z.sum(axis=0);weights=.8*weights+.2/M
    p=models(x,counts);return (p*weights[:,:,None]).sum(0),weights,np.array(records)
SUM=np.indices((10,10,10)).sum(0)
@st.cache_data
def calculate(x):return train(x)

with st.sidebar:
    st.write('PostgreSQL' if URL else 'SQLite 本地模式（云端重启可能丢失）')
    auto=st.checkbox('自动同步',True); n=st.number_input('首次同步期数',100,5000,5000,100)
    if st.button('同步历史开奖'):
        try:st.success(f'读取 {sync(n)} 条不同期号')
        except Exception as e:st.error(str(e))
    upload=st.file_uploader('备用 CSV 导入（issue,a,b,c,stamp）',type=['csv'])
    if upload and st.button('导入 CSV'):
        rows=parse(pd.read_csv(upload).to_dict('records'));save(rows);st.success(f'导入 {len(rows)} 条有效记录')

@st.fragment(run_every=5 if auto else None)
def live():
    if auto:
        try:
            save(fetch('/draws/latest'))
            now=datetime.now().timestamp()
            if now-st.session_state.get('last_sync',0)>=210:
                sync(100);st.session_state.last_sync=now
        except Exception as e:st.warning(f'接口同步失败：{e}')
    d=history()
    if len(d)<100:st.info('请先同步或导入至少100期历史数据。');return
    last=d.iloc[-1]; issue=int(last.issue)+1; x=d[['a','b','c']].to_numpy(dtype=int)
    st.write(f'最新第 {int(last.issue)} 期：{last.a} + {last.b} + {last.c} = {sum(x[-1])}');st.write(f'目标期号：{issue}｜历史 {len(d)} 期｜北京时间 {datetime.now(TZ):%Y-%m-%d %H:%M:%S}')
    try:
        dt=pd.to_datetime(last.stamp,utc=True);remain=max(0,int((dt+pd.Timedelta(seconds=210)-pd.Timestamp.now(tz='UTC')).total_seconds()));st.caption(f'估算倒计时 {remain//60:02d}:{remain%60:02d}（无时区时间按UTC解析）')
    except Exception:pass
    p,w,rec=calculate(x);joint=np.einsum('i,j,k->ijk',*p);big=float(joint[SUM>=14].sum());odd=float(joint[SUM%2==1].sum())
    order=np.argsort(-p,axis=1); payload={'seven':order[:,:7].tolist(),'eight':order[:,:8].tolist(),'p':p.tolist(),'big':big,'odd':odd}
    with E.begin() as c:
        c.execute(text('INSERT INTO v700_predictions(issue,created,payload) VALUES(:i,:t,:p) ON CONFLICT(issue) DO NOTHING'),dict(i=issue,t=datetime.now(TZ).isoformat(),p=json.dumps(payload)))
        pending=c.execute(text('SELECT issue FROM v700_predictions WHERE actual IS NULL')).scalars().all(); lookup={int(r.issue):[int(r.a),int(r.b),int(r.c)] for r in d.itertuples()}
        for i in pending:
            if i in lookup:c.execute(text('UPDATE v700_predictions SET actual=:a WHERE issue=:i AND actual IS NULL'),dict(a=json.dumps(lookup[i]),i=i))
        saved=c.execute(text('SELECT payload FROM v700_predictions WHERE issue=:i'),dict(i=issue)).scalar_one();payload=json.loads(saved)
        logs=c.execute(text('SELECT issue,created,payload,actual FROM v700_predictions ORDER BY issue DESC LIMIT 200')).all()
    p=np.array(payload['p']);st.subheader('三球七码 / 八码')
    for j,col in enumerate(st.columns(3)):
        with col:
            st.markdown(f'**{chr(65+j)}球**')
            for key,label in [('seven','七码'),('eight','八码')]:
                k=payload[key][j];st.write(label+'：'+' '.join(map(str,sorted(k))));st.caption(f'模型覆盖概率 {p[j,k].sum():.1%}')
    for title,q,a,b in [('大小',payload['big'],'大','小'),('单双',payload['odd'],'单','双')]:
        st.subheader(title);c1,c2=st.columns(2);c1.metric(a,f'{q:.1%}');c2.metric(b,f'{1-q:.1%}');st.caption('暂不推荐' if abs(q-.5)<.03 else '模型倾向（未经未来验证）：'+(a if q>.5 else b))
    if len(rec):
        st.subheader('历史逐期回测（第101条起）');st.dataframe(pd.DataFrame({'指标':['A七码','B七码','C七码','A八码','B八码','C八码','七码全中','八码全中','大小','单双'],'命中率':rec.mean(0),'独立均匀基准':[.7]*3+[.8]*3+[.343,.512,.5,.5]}))
    st.subheader('模型权重');st.dataframe(pd.DataFrame(w,index=NAMES,columns=list('ABC')))
    st.subheader('实际预测记录（最近200期）'); rows=[]
    for i,t,s,a in logs:
        z=json.loads(s);y=json.loads(a) if a else None; r={'期号':i,'时间':t,'实际':str(y) if y else '等待开奖'}
        for j in range(3):
            for k in ('seven','eight'):r[f'{chr(65+j)}{k}']='等待开奖' if y is None else ('正确' if y[j] in z[k][j] else '错误')
        for k,label,truth in [('big','大小',sum(y)>=14 if y else None),('odd','单双',sum(y)%2==1 if y else None)]:r[label]='等待开奖' if y is None else ('正确' if (z[k]>=.5)==truth else '错误')
        rows.append(r)
    st.dataframe(pd.DataFrame(rows))
live()
