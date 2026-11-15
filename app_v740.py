import os, json, math, time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title='Canada28 Matrix AI V7.5.1',layout='wide')
st.title('Canada28 Matrix AI V7.5.1')
st.caption('统计研究：随机开奖不能可靠预测；候选覆盖率不等于盈利概率。数据来自 [pc28.world](https://pc28.world)。')
TZ=timezone(timedelta(hours=8)); BASE='https://pc28.world/api/v1'
URL=os.getenv('DATABASE_URL','')
try: URL=st.secrets.get('DATABASE_URL',URL)
except Exception: pass
@st.cache_resource
def db(url):
    e=create_engine(url.replace('postgres://','postgresql+psycopg://',1).replace('postgresql://','postgresql+psycopg://',1) if url else 'sqlite:///v720_local.db')
    with e.begin() as c:
        c.execute(text('CREATE TABLE IF NOT EXISTS v720_draws (issue BIGINT PRIMARY KEY, stamp TEXT, a INTEGER, b INTEGER, c INTEGER)'))
        c.execute(text('CREATE TABLE IF NOT EXISTS v720_predictions (issue BIGINT PRIMARY KEY, created TEXT, payload TEXT, actual TEXT)'))
        c.execute(text('CREATE TABLE IF NOT EXISTS v750_back (issue BIGINT PRIMARY KEY, payload TEXT NOT NULL, actual TEXT NOT NULL)'))
        c.execute(text('CREATE TABLE IF NOT EXISTS v750_live (issue BIGINT PRIMARY KEY, created TEXT, payload TEXT NOT NULL, actual TEXT)'))
        c.execute(text('CREATE TABLE IF NOT EXISTS v750_state (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)'))
    return e
E=db(URL)

def parse(raw):
    if isinstance(raw,dict):
        for k in ('data','draw','draws','results','items','records','list'):
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
            try:s=str(get('draw_time','time','开奖时间','timestamp','open_time','created_at','stamp'))
            except Exception:s=''
            out.append(dict(issue=i,stamp=s,a=nums[0],b=nums[1],c=nums[2]))
        except Exception:continue
    return out

def save(rows):
    with E.begin() as c:
        if rows:c.execute(text('INSERT INTO v720_draws(issue,stamp,a,b,c) VALUES(:issue,:stamp,:a,:b,:c) ON CONFLICT(issue) DO NOTHING'),rows)

def fetch(path,params=None):
    r=requests.get(BASE+path,params=params,timeout=15);r.raise_for_status();return parse(r.json())

def sync(n):
    seen=set(); before=None
    while len(seen)<n:
        params={'limit':min(200,n-len(seen))}
        if before is not None:params['before']=before
        rows=fetch('/draws',params)
        new=[r for r in rows if r['issue'] not in seen and (before is None or r['issue']<before)]
        if not new:
            if len(seen)<n:raise RuntimeError(f'历史同步仅取得 {len(seen)} / {n} 期，接口未返回更多有效记录。已保存已取得数据。')
            break
        save(new);seen.update(r['issue'] for r in new)
        before=min(r['issue'] for r in new)
        if len(seen)<n:time.sleep(1.1)
    return len(seen)

def history():
    with E.connect() as c:return pd.read_sql(text('SELECT * FROM v720_draws ORDER BY issue DESC'),c).sort_values('issue').reset_index(drop=True)


# Independent task weights: ball x task x model.
NAMES=['均匀基准','加法','减法','乘法','除法','横向','纵向','正斜向','反斜向','反纵向','近期频率','贝叶斯','代数','几何','离散微积分','状态转移']
TASKS=['七码','八码','大小','单双']
M=len(NAMES)
def features(x):
    u,v,w=x[-1],x[-2],x[-3]
    return np.stack([u,(u+v)%10,(u-v)%10,(u*v)%10,np.floor(u/(v+1)).astype(int)%10,(u.sum()-u)%10,v,(u+np.roll(v,1)+np.roll(w,2))%10,(u+np.roll(v,-1)+np.roll(w,-2))%10,(w-v)%10,u,u,(2*u-v)%10,np.floor(np.sqrt((u-v)**2+(v-w)**2)).astype(int)%10,(u-2*v+w)%10,u])
def models(x,counts):
    f=features(x); p=(counts+3)/(counts.sum(-1,keepdims=True)+30)
    out=p[np.arange(M)[:,None],np.arange(3),f];out[0]=.1
    for m,win in [(10,100),(11,500)]:
        z=x[-win:];out[m]=np.stack([(np.bincount(z[:,j],minlength=10)+20)/(len(z)+200) for j in range(3)])
    return out
def predict(p,w):
    mix=np.einsum('jkm,mjd->jkd',w,p); result=[]
    for j in range(3):
        r=[]
        for k in range(4):
            q=mix[j,k]; order=np.argsort(-q,kind='stable')
            if k<2:
                a=sorted(order[:7+k].tolist());prob=float(q[a].sum())
            else:
                prob=float(q[5:].sum() if k==2 else q[1::2].sum());a=int(prob>=.5)
            r.append(dict(pick=a,prob=prob))
        result.append(r)
    return result
def truth(y,k):return int(y>=5) if k==2 else int(y%2) if k==3 else int(y)
def hit(p,y,k):return y in p if k<2 else p==truth(y,k)

# Model availability: no future observations, and no crossing gaps for lag features.
NEED=np.array([0,2,2,2,2,1,2,3,3,3,1,1,2,3,3,1])
def available(issues):
    n=0
    for i in range(len(issues)-1,-1,-1):
        if i<len(issues)-1 and issues[i+1]!=issues[i]+1:break
        n+=1
        if n==3:break
    return NEED<=n

def model_probs(x,counts,issues):
    avail=available(issues)
    padded=np.pad(x,((max(0,3-len(x)),0),(0,0)),constant_values=0)
    f=features(padded)
    p=(counts+3)/(counts.sum(-1,keepdims=True)+30)
    out=p[np.arange(M)[:,None],np.arange(3),f].copy();out[0]=.1
    for m,win in [(10,100),(11,500)]:
        z=x[-win:];out[m]=np.stack([(np.bincount(z[:,j],minlength=10)+20)/(len(z)+200) for j in range(3)])
    return out,avail,f

def weights(loss):
    a=np.exp(-12*(loss-loss.min(-1,keepdims=True)));return .8*a/a.sum(-1,keepdims=True)+.2/M

def pack(p,avail,w):
    # 17 x 3 x 4 picks, compact per issue; probability vectors retained for audit.
    order=np.argsort(-p,axis=-1,kind='stable')
    picks=[]
    for m in range(M):
        if not avail[m]:picks.append(None);continue
        picks.append([[sorted(order[m,b,:7].tolist()),sorted(order[m,b,:8].tolist()),int(p[m,b,5:].sum()>=.5),int(p[m,b,1::2].sum()>=.5)] for b in range(3)])
    ww=w*avail[None,None,:];ww=ww/ww.sum(-1,keepdims=True)
    mix=np.einsum('bkm,mbd->bkd',ww,p)
    pred=predict(p,ww)
    picks.append([[r['pick'] for r in rr] for rr in pred])
    return {'picks':picks,'p':p.round(8).tolist(),'mix':mix.round(8).tolist(),'w':ww.round(8).tolist()}

def update_loss(loss,pack0,y):
    for m,pp in enumerate(pack0['picks'][:M]):
        if pp is None:continue
        for b in range(3):
            for k in range(4):
                loss[b,k,m]=.995*loss[b,k,m]+.005*(not hit(pp[b][k],y[b],k))

def initial():return np.zeros((M,3,10,10)),np.zeros((3,4,M)),np.full((3,4,M),1/M)

def replay(d,progress=None):
    # Checkpoint includes a fingerprint of the entire processed prefix. Rebuild if earlier history changes.
    import hashlib
    x=d[['a','b','c']].to_numpy(dtype=int); issues=d.issue.to_numpy(dtype=int)
    def digest(t):return hashlib.sha256(np.column_stack((issues[:t],x[:t])).astype('<i8').tobytes()).hexdigest()
    counts,loss,w=initial(); start=0
    with E.connect() as c:r=c.execute(text('SELECT payload FROM v750_state WHERE id=1')).scalar()
    if r:
        a=json.loads(r);t=a['n']
        if t<=len(x) and a['hash']==digest(t):
            start=t;counts=np.array(a['counts']);loss=np.array(a['loss']);w=np.array(a['w'])
    if start==0:
        with E.begin() as c:c.execute(text('DELETE FROM v750_back'));c.execute(text('DELETE FROM v750_state'))
    batch=[]
    for t in range(start,len(x)):
        p,av,f=model_probs(x[:t],counts,issues[:t]); y=x[t]
        if t and issues[t]!=issues[t-1]+1:av=NEED==0
        pp=pack(p,av,w);update_loss(loss,pp,y)
        if (t+1)%50==0:w=weights(loss)
        for m in np.flatnonzero(av):counts[m,np.arange(3),f[m],y]+=1
        batch.append(dict(i=int(issues[t]),p=json.dumps(pp,separators=(',',':')),a=json.dumps(y.tolist())))
        if len(batch)>=100 or t==len(x)-1:
            state=json.dumps(dict(n=t+1,hash=digest(t+1),counts=counts.tolist(),loss=loss.tolist(),w=w.tolist()))
            with E.begin() as c:
                c.execute(text('INSERT INTO v750_back(issue,payload,actual) VALUES(:i,:p,:a) ON CONFLICT(issue) DO UPDATE SET payload=excluded.payload,actual=excluded.actual'),batch)
                c.execute(text('INSERT INTO v750_state(id,payload) VALUES(1,:p) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload'),dict(p=state))
            batch=[]
            if progress:progress.progress((t+1)/len(x),text=f'历史逐期回测 {t+1}/{len(x)}')
    return counts,w

def load_records(table,limit=None):
    sql=f'SELECT issue,payload,actual FROM {table} ORDER BY issue DESC'+(' LIMIT :n' if limit else '')
    with E.connect() as c:rows=c.execute(text(sql),dict(n=limit) if limit else {}).all()
    return [(int(i),json.loads(p),json.loads(a) if a else None) for i,p,a in reversed(rows)]

def label(p,k):return ' '.join(map(str,p)) if k<2 else (['小','大'] if k==2 else ['双','单'])[p]
def detail(rec,m,b,k):
    rows=[]
    for i,p,y in rec:
        pp=p['picks'][m];a=None if pp is None else pp[b][k];v=None if y is None else y[b]
        rows.append((i,'—' if a is None else label(a,k),'—' if v is None else v,'未具备条件' if a is None else '⏳' if v is None else '✅' if hit(a,v,k) else '❌'))
    return pd.DataFrame(rows,columns=['期号','预测','实际','结果'])
def runs(d):
    out=[];r=None;last=None
    for i,s in zip(d['期号'],d['结果']):
        if s not in ('✅','❌'):
            if r:out.append(r);r=None
        elif r and s==r['结果'] and i==last+1:r['结束']=i;r['长度']+=1
        else:
            if r:out.append(r)
            r=dict(结果=s,开始=i,结束=i,长度=1)
        last=i
    if r:out.append(r)
    return out

def stats(d):
    h=int((d['结果']=='✅').sum());e=int((d['结果']=='❌').sum());r=runs(d)
    settled=d[d['结果'].isin(['✅','❌'])];last=settled.iloc[-1]['期号'] if len(settled) else None
    return [h+e,h,e,f'{h/(h+e):.1%}' if h+e else '—',max([a['长度'] for a in r if a['结果']=='✅'] or [0]),max([a['长度'] for a in r if a['结果']=='❌'] or [0]),r[-1]['结果']+str(r[-1]['长度']) if r and r[-1]['结束']==last else '—']

with st.sidebar:
    st.write('PostgreSQL 已连接' if URL else 'SQLite 本地模式：云端重启可能丢失')
    auto=st.checkbox('自动同步',True);n=st.number_input('历史同步目标（最近期数）',100,5000,5000,100)
    zone=st.selectbox('无时区开奖时间解释',['北京时间','UTC'])
    if st.button('同步历史开奖'):
        try:st.success(f'读取 {sync(n)} 条不同期号')
        except Exception as e:st.error(str(e))
    upload=st.file_uploader('备用 CSV（issue,a,b,c,stamp）',type=['csv'])
    if upload and st.button('导入 CSV'):save(parse(pd.read_csv(upload).to_dict('records')))
    source=st.radio('统计来源',['历史逐期回测','在线预测']);window=st.selectbox('统计区间',[20,50,100,500,1000,5000,'全部'],index=6)

@st.fragment(run_every=5 if auto else None)
def live():
    if auto:
        try:
            save(fetch('/draws/latest'));now=time.time()
            if now-st.session_state.get('last_sync',0)>=210:
                st.session_state.last_sync=now
                sync(n if len(history())<n else 100)
        except Exception as e:st.warning(f'同步未完成：{e}')
    d=history()
    if d.empty:st.info('暂无历史数据，请同步或导入。');return
    x=d[['a','b','c']].to_numpy(dtype=int);issues=d.issue.to_numpy(dtype=int);last=d.iloc[-1];issue=int(last.issue)+1
    st.write(f'最新第 {int(last.issue)} 期：{last.a} + {last.b} + {last.c} = {sum(x[-1])}')
    st.caption(f'目标 {issue}｜历史 {len(d)} 期｜最早可用期号 {issues[0]}｜北京时间 {datetime.now(TZ):%Y-%m-%d %H:%M:%S}')
    remain=None
    try:
        dt=pd.Timestamp(last.stamp);dt=dt.tz_localize('Asia/Shanghai' if zone=='北京时间' else 'UTC') if dt.tzinfo is None else dt
        remain=(dt+pd.Timedelta(seconds=210)-pd.Timestamp.now(tz='UTC')).total_seconds()
        st.caption(f'预计倒计时 {int(remain)//60:02d}:{int(remain)%60:02d}' if 0<remain<=210 else '等待新开奖或检查时间设置')
    except Exception:st.caption('开奖时间无法解析')
    progress=st.progress(0,text='检查历史回测进度')
    counts,w=replay(d,progress);progress.empty()
    with E.begin() as c:
        pending=c.execute(text('SELECT issue FROM v750_live WHERE actual IS NULL')).scalars().all()
        for i in pending:
            row=c.execute(text('SELECT a,b,c FROM v720_draws WHERE issue=:i'),dict(i=i)).first()
            if row:c.execute(text('UPDATE v750_live SET actual=:a WHERE issue=:i AND actual IS NULL'),dict(a=json.dumps(list(row)),i=i))
    online=load_records('v750_live');settled=[r for r in online if r[2] is not None]
    # Live calibration uses completed 50-issue blocks only, independently of historical records.
    if len(settled)>=50:
        loss=np.zeros((3,4,M))
        for _,pp,y in settled[:len(settled)//50*50]:update_loss(loss,pp,y)
        w=weights(loss)
    p,av,f=model_probs(x,counts,issues);pp=pack(p,av,w)
    # Recheck clock and latest draw after computation; never intentionally lock a past target.
    if auto and remain is not None:
        try:
            fresh=fetch('/draws/latest');save(fresh)
            remaining=(dt+pd.Timedelta(seconds=210)-pd.Timestamp.now(tz='UTC')).total_seconds()
            if fresh and max(r['issue'] for r in fresh)==issue-1 and 0<remaining<=210:
                with E.begin() as c:c.execute(text('INSERT INTO v750_live(issue,created,payload) VALUES(:i,:t,:p) ON CONFLICT(issue) DO NOTHING'),dict(i=issue,t=datetime.now(TZ).isoformat(),p=json.dumps(pp)))
        except Exception as e:st.warning(f'在线记录未锁定：{e}')
    online=load_records('v750_live');saved=next((p for i,p,y in online if i==issue),None)
    if saved:pp=saved
    st.caption('已锁定在线预测' if saved else '研究预览：尚未锁定在线预测')
    rec=load_records('v750_back' if source=='历史逐期回测' else 'v750_live',None if window=='全部' else int(window))
    st.subheader('ABC 三球 · 12项独立综合推演')
    names=NAMES+['AI综合']
    for b in range(3):
        st.header(f'{chr(65+b)}球')
        cols=st.columns(4)
        for k,t in enumerate(TASKS):
            a=pp['picks'][M][b][k];q=np.array(pp['mix'][b][k])
            prob=float(q[a].sum()) if k<2 else float(q[5:].sum() if k==2 else q[1::2].sum())
            with cols[k]:
                st.metric(t,label(a,k))
                st.caption(f'模型覆盖概率 {prob:.1%}' if k<2 else (f'大 {prob:.1%} / 小 {1-prob:.1%}' if k==2 else f'单 {prob:.1%} / 双 {1-prob:.1%}'))
        summary=[]; distributions=[]
        for k,t in enumerate(TASKS):
            df=detail(rec,M,b,k);summary.append([t,*stats(df)])
            rr=runs(df)
            for status in ['❌','✅']:
                lengths={}
                for r in rr:
                    if r['结果']==status:lengths[r['长度']]=lengths.get(r['长度'],0)+1
                distributions.append([t,status,' / '.join(f'{l}期:{v}次' for l,v in sorted(lengths.items())) or '—'])
        st.markdown('**综合预测历史成绩**')
        st.dataframe(pd.DataFrame(summary,columns=['项目','已核对','正确','错误','命中率','最高连胜','最高连错','当前']),hide_index=True)
        st.markdown('**历史连胜连错次数**')
        st.dataframe(pd.DataFrame(distributions,columns=['项目','状态','连续长度及次数']),hide_index=True)
        # All four independent histories appear together, newest first.
        combined=pd.DataFrame({'期号':[i for i,_,_ in rec]})
        for k,t in enumerate(TASKS):
            df=detail(rec,M,b,k)
            combined[t+'预测']=df['预测'];combined[t+'结果']=df['结果']
            if k==0:combined['实际号码']=df['实际']
        st.markdown('**逐期综合预测记录**')
        st.dataframe(combined.iloc[::-1],hide_index=True,height=320)
        st.download_button(f'下载{chr(65+b)}球历史 CSV',combined.to_csv(index=False).encode('utf-8-sig'),file_name=f'v751_{chr(65+b)}.csv',key=f'export_{b}')
    st.subheader('历史连错详细记录')
    b=st.selectbox('明细球号',range(3),format_func=lambda i:chr(65+i));k=st.selectbox('明细项目',range(4),format_func=lambda i:TASKS[i]);df=detail(rec,M,b,k)
    rr=pd.DataFrame(runs(df),columns=['结果','开始','结束','长度']);st.dataframe(rr.iloc[::-1],hide_index=True)
    st.dataframe(df.iloc[::-1],hide_index=True)
    with st.expander('独立数学模型历史成绩（可选查看）'):
        mb=st.selectbox('模型明细球号',range(3),format_func=lambda i:chr(65+i));mk=st.selectbox('模型明细项目',range(4),format_func=lambda i:TASKS[i]);mm=st.selectbox('独立模型',range(M),format_func=lambda i:NAMES[i])
        md=detail(rec,mm,mb,mk)
        st.dataframe(pd.DataFrame([[NAMES[mm],*stats(md)]],columns=['模型','已核对','正确','错误','命中率','最高连胜','最高连错','当前']),hide_index=True)
        st.dataframe(md.iloc[::-1],hide_index=True)
        st.dataframe(pd.DataFrame(runs(md),columns=['结果','开始','结束','长度']).iloc[::-1],hide_index=True)
    with st.expander('数学引擎：全部模型概率及权重'):
        rows=[]
        for b in range(3):
            for k in range(4):
                for m,name in enumerate(NAMES):rows.append([chr(65+b),TASKS[k],name,pp['w'][b][k][m],*pp['p'][m][b]])
        st.dataframe(pd.DataFrame(rows,columns=['球','项目','模型','权重',*map(str,range(10))]),hide_index=True)
    st.caption('第一期仅理论基准可用；缺少必要历史的模型不计成绩。历史与在线独立保存。当前统计窗口可能截断连续记录。')
live()
