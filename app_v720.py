import os, json, math
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title='Canada28 Matrix AI V7.2',layout='wide')
st.title('Canada28 Matrix AI V7.2')
st.caption('统计研究：随机开奖不能可靠预测；候选覆盖率不等于盈利概率。')
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
    seen=set()
    for page in range(1, min(100,math.ceil(n/100)+2)+1):
        rows=fetch('/draws',{'limit':100,'page':page}); new=[r for r in rows if r['issue'] not in seen]
        if not new:break
        save(new);seen.update(r['issue'] for r in new)
        if len(seen)>=n:break
    return len(seen)

def history():
    with E.connect() as c:return pd.read_sql(text('SELECT * FROM v720_draws ORDER BY issue DESC LIMIT 5000'),c).sort_values('issue').reset_index(drop=True)


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
def train(x,issues):
    counts=np.zeros((M,3,10,10)); loss=np.zeros((3,4,M)); w=np.full((3,4,M),1/M); records=[]; modelhits=np.zeros((3,4,M)); evaluated=0
    for t in range(3,len(x)):
        if not np.all(np.diff(issues[t-3:t+1])==1):continue
        z=x[:t];y=x[t];p=models(z,counts);f=features(z); pred=predict(p,w)
        order=np.argsort(-p,axis=-1,kind='stable'); big=p[:,:,5:].sum(-1);odd=p[:,:,1::2].sum(-1)
        h=np.stack([(order[:,:,:7]==y[None,:,None]).any(-1),(order[:,:,:8]==y[None,:,None]).any(-1),(big>=.5)==(y>=5),(odd>=.5)==(y%2==1)],axis=1).transpose(2,1,0)
        if t>=100:
            records.append((int(issues[t]),pred,y.tolist()));modelhits+=h;evaluated+=1
            loss=.995*loss+.005*(1-h.astype(float))
            if evaluated%50==0:
                a=np.exp(-12*(loss-loss.min(-1,keepdims=True)));w=.8*a/a.sum(-1,keepdims=True)+.2/M
        for m in range(M):counts[m,np.arange(3),f[m],y]+=1
    return predict(models(x,counts),w),w,records,modelhits/max(1,evaluated)
@st.cache_data
def calculate(x,issues):return train(x,issues)
def label(p,k):
    return ' '.join(map(str,p)) if k<2 else (['小','大'] if k==2 else ['双','单'])[p]
def detail(records,j,k):
    rows=[]
    for i,p,y in records:
        a=p[j][k]['pick'];v=None if y is None else int(y[j]);h=None if v is None else hit(a,v,k)
        rows.append(dict(期号=i,预测=label(a,k),实际='—' if v is None else str(v),结果='⏳' if h is None else '✅' if h else '❌'))
    return pd.DataFrame(rows,columns=['期号','预测','实际','结果'])
def runs(d):
    out=[]; last=None; r=None
    for i,s in zip(d['期号'],d['结果']):
        if s not in ('✅','❌'):
            if r:out.append(r);r=None
        elif r and s==r['结果'] and i==last+1:r['结束']=i;r['长度']+=1
        else:
            if r:out.append(r)
            r={'结果':s,'开始':i,'结束':i,'长度':1}
        last=i
    if r:out.append(r)
    return out
def stats(d):
    n=int(d['结果'].isin(['✅','❌']).sum()); h=int((d['结果']=='✅').sum());r=runs(d)
    return n,h,n-h,h/n if n else 0,max([a['长度'] for a in r if a['结果']=='✅'] or [0]),max([a['长度'] for a in r if a['结果']=='❌'] or [0]),(r[-1]['结果']+str(r[-1]['长度']) if r and len(d) and r[-1]['结束']==d.iloc[-1]['期号'] else '—')
with st.sidebar:
    st.write('PostgreSQL 已连接' if URL else 'SQLite 本地模式：云端重启可能丢失')
    auto=st.checkbox('自动同步',True);n=st.number_input('历史同步期数',100,5000,5000,100)
    zone=st.selectbox('无时区开奖时间解释',['北京时间','UTC'])
    if st.button('同步历史开奖'):
        try:st.success(f'读取 {sync(n)} 条不同期号')
        except Exception as e:st.error(str(e))
    upload=st.file_uploader('备用 CSV（issue,a,b,c,stamp）',type=['csv'])
    if upload and st.button('导入 CSV'):save(parse(pd.read_csv(upload).to_dict('records')))
    page=st.radio('页面',['实时推演','历史预测','历史统计','连胜连错','数学引擎'])
    source=st.radio('统计来源',['历史逐期回测','在线预测'])
    window=st.selectbox('统计区间',[20,50,100,500,1000,5000,'全部'],index=6)
    ball=st.selectbox('球号',list('ABC'));task=st.selectbox('项目',TASKS)
@st.fragment(run_every=5 if auto else None)
def live():
    if auto:
        try:
            save(fetch('/draws/latest'));now=datetime.now().timestamp()
            if now-st.session_state.get('last_sync',0)>=210:
                sync(n if len(history())<n else 100);st.session_state.last_sync=now
        except Exception as e:st.warning(f'同步失败：{e}')
    d=history()
    if len(d)<101:st.info('至少需要101期历史，请同步或导入。');return
    x=d[['a','b','c']].to_numpy(dtype=int);issues=d.issue.to_numpy(dtype=int);last=d.iloc[-1];issue=int(last.issue)+1
    pred,w,back,mh=calculate(x,issues)
    st.write(f'最新第 {int(last.issue)} 期：{last.a} + {last.b} + {last.c} = {sum(x[-1])}');st.caption(f'目标 {issue}｜历史 {len(d)} 期｜北京时间 {datetime.now(TZ):%Y-%m-%d %H:%M:%S}')
    remain=None
    try:
        dt=pd.Timestamp(last.stamp);dt=dt.tz_localize('Asia/Shanghai' if zone=='北京时间' else 'UTC') if dt.tzinfo is None else dt
        remain=(dt+pd.Timedelta(seconds=210)-pd.Timestamp.now(tz='UTC')).total_seconds()
        st.caption(f'预计倒计时 {int(remain)//60:02d}:{int(remain)%60:02d}' if 0<remain<=210 else '等待新开奖或检查时间设置')
    except Exception:st.caption('开奖时间无法解析')
    with E.begin() as c:
        # Only save live predictions within the estimated pre-draw window.
        if auto and remain is not None and 0<remain<=210 and np.all(np.diff(issues[-3:])==1):
            c.execute(text('INSERT INTO v720_predictions(issue,created,payload) VALUES(:i,:t,:p) ON CONFLICT(issue) DO NOTHING'),dict(i=issue,t=datetime.now(TZ).isoformat(),p=json.dumps(pred)))
        c.execute(text("UPDATE v720_predictions SET actual=NULL WHERE actual IS NULL")) if False else None
        pending=c.execute(text('SELECT issue FROM v720_predictions WHERE actual IS NULL')).scalars().all()
        for i in pending:
            row=c.execute(text('SELECT a,b,c FROM v720_draws WHERE issue=:i'),dict(i=i)).first()
            if row:c.execute(text('UPDATE v720_predictions SET actual=:a WHERE issue=:i AND actual IS NULL'),dict(a=json.dumps(list(row)),i=i))
        logs=c.execute(text('SELECT issue,payload,actual FROM v720_predictions ORDER BY issue')).all()
    online=[(int(i),json.loads(p),json.loads(a) if a else None) for i,p,a in logs]
    saved=next((p for i,p,y in online if i==issue),None)
    if saved is not None:pred=saved
    rec=back if source=='历史逐期回测' else online
    if window!='全部':rec=rec[-int(window):]
    j=list('ABC').index(ball);k=TASKS.index(task);df=detail(rec,j,k)
    if page=='实时推演':
        st.caption('已锁定在线预测' if saved is not None else '研究预览：尚未锁定在线预测')
        for b,col in enumerate(st.columns(3)):
            with col:
                st.subheader(f'{chr(65+b)}球')
                for k0,t in enumerate(TASKS):
                    r=pred[b][k0];st.write(f'**{t}：{label(r["pick"],k0)}**');q=r['prob'];st.caption(f'模型覆盖概率 {q:.1%}' if k0<2 else f'{label(1,k0)} {q:.1%} / {label(0,k0)} {1-q:.1%}')
    elif page=='历史预测':st.dataframe(df.iloc[::-1],hide_index=True)
    elif page=='历史统计':
        rows=[]
        for b in range(3):
            for k0,t in enumerate(TASKS):
                a=stats(detail(rec,b,k0));rows.append([chr(65+b),t,*a[:3],f'{a[3]:.1%}' if a[0] else '—',a[4],a[5],a[6]])
        st.dataframe(pd.DataFrame(rows,columns=['球','项目','已核对','正确','错误','命中率','最高连胜','最高连错','当前']),hide_index=True)
        st.subheader('三球联合命中')
        rows=[]
        for k0,t in enumerate(TASKS):
            counts=[0]*4
            for i,p,y in rec:
                if y is not None:counts[sum(hit(p[b][k0]['pick'],y[b],k0) for b in range(3))]+=1
            rows.append([t,*counts])
        st.dataframe(pd.DataFrame(rows,columns=['项目','全错','中一球','中两球','全中']),hide_index=True)
    elif page=='连胜连错':
        a=stats(df);st.write(f'已核对 {a[0]}｜正确 {a[1]}｜错误 {a[2]}｜当前 {a[6]}｜最高连胜 {a[4]}｜最高连错 {a[5]}')
        r=pd.DataFrame(runs(df),columns=['结果','开始','结束','长度']);st.caption('完整区段计数；区间边缘可能截断，缺期或等待开奖中断连续记录。')
        if len(r):
            st.dataframe(r.groupby(['长度','结果']).size().unstack(fill_value=0),use_container_width=True)
            st.dataframe(r.sort_values('开始',ascending=False),hide_index=True)
            for a in r[r['结果']=='❌'].sort_values('开始',ascending=False).head(100).to_dict('records'):
                with st.expander(f"{a['长度']}连错：{a['开始']}–{a['结束']}"):st.dataframe(df[df['期号'].between(a['开始'],a['结束'])],hide_index=True)
        else:st.info('暂无已核对记录')
    else:
        st.dataframe(pd.DataFrame({'公式':NAMES,'权重':w[j,k],'逐期回测命中率':mh[j,k]}),hide_index=True)
        st.caption('公式特征经条件频率映射为数字概率；各任务按独立命中表现每50个有效回测目标更新权重。')
live()
