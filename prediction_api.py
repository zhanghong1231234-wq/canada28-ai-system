"""Read-only API for V7.6/V7.7 locked predictions. Run separately from Streamlit."""
import os, json, hmac
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Header, HTTPException
from sqlalchemy import create_engine, text

URL=os.environ.get('DATABASE_URL','')
KEY=os.environ.get('PREDICTION_API_KEY','')
if not URL or not KEY: raise RuntimeError('DATABASE_URL and PREDICTION_API_KEY required')
engine=create_engine(URL.replace('postgres://','postgresql+psycopg://',1).replace('postgresql://','postgresql+psycopg://',1),pool_pre_ping=True)
app=FastAPI(title='Canada28 read-only prediction API')
TASKS=['seven','eight','size','parity']

def summarize(payload, records):
    result=[]
    for b in range(3):
        for k,task in enumerate(TASKS):
            pick=payload['picks'][-1][b][k]; q=payload['mix'][b][k]
            prob=sum(q[d] for d in pick) if k<2 else sum(q[d] for d in range(10) if (int(d>=5) if k==2 else d%2)==pick)
            hits=0; n=0
            for p,y in records:
                a=p['picks'][-1][b][k]; v=y[b]; n+=1
                hits+=int(v in a if k<2 else a==(int(v>=5) if k==2 else v%2))
            result.append(dict(ball='ABC'[b],task=task,pick=pick,model_probability=prob,settled=n,correct=hits,incorrect=n-hits,hit_rate=hits/n if n else None))
    return result

@app.get('/v1/predictions/latest')
def latest(authorization: str = Header(default='')):
    if not hmac.compare_digest(authorization,'Bearer '+KEY): raise HTTPException(401,'Unauthorized')
    with engine.connect() as c:
        draw=c.execute(text('SELECT issue,stamp FROM v720_draws ORDER BY issue DESC LIMIT 1')).first()
        if not draw: raise HTTPException(503,'No draw data')
        target=int(draw[0])+1
        row=c.execute(text('SELECT created,payload FROM v760_live WHERE issue=:i AND actual IS NULL'),{'i':target}).first()
        if not row: raise HTTPException(503,'No locked prediction for next issue')
        records=c.execute(text('SELECT payload,actual FROM v760_live WHERE actual IS NOT NULL AND issue<:i ORDER BY issue'),{'i':target}).all()
    try:
        dt=datetime.fromisoformat(str(draw[1]).replace('Z','+00:00'))
        if dt.tzinfo is None: dt=dt.replace(tzinfo=timezone(timedelta(hours=8)))
        # Informational estimated time only. Prediction validity is determined by
        # current source_issue + 1 and the existence of an unsettled locked row.
        expires=dt+timedelta(seconds=210)
    except (ValueError,TypeError): raise HTTPException(503,'Invalid draw timestamp')
    payload=json.loads(row[1])
    balls={
        ball:{
            'eight_numbers':list(payload['picks'][-1][b][1]),
            'number_probabilities':[float(v) for v in payload['mix'][b][1]]
        }
        for b,ball in enumerate(('A','B','C'))
    }
    return dict(system='Canada28 Matrix AI V7.8',issue=target,source_issue=int(draw[0]),created=row[0],estimated_draw_at=expires.isoformat(),locked=True,balls=balls,tasks=summarize(payload,[(json.loads(p),json.loads(y)) for p,y in records]))
