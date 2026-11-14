import pandas as pd, joblib, json,hashlib,sys,sklearn,numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import argparse
parser=argparse.ArgumentParser(description='Evaluate saved model on the reconstructed baseline split without training')
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--output',type=Path,default=None)
args=parser.parse_args()
root=args.root.resolve()
p=next(root.glob('datasets/**/*.csv'));data=pd.read_csv(p)
df=data[['content','score']].dropna().copy();df['content']=df.content.str.strip();df=df[df.content.ne('')&df.score.isin([1,2,3,4,5])].copy()
df['sentiment']=df.score.map({1:'negative',2:'negative',3:'neutral',4:'positive',5:'positive'})
df['text_key']=df.content.str.casefold().str.replace(r'\s+',' ',regex=True)
df=df[df.text_key!='#name?'].copy();counts=df.groupby('text_key').sentiment.nunique();df=df[~df.text_key.isin(counts[counts>1].index)].drop_duplicates('text_key').copy()
train,rem=train_test_split(df,test_size=.30,random_state=42,stratify=df.sentiment)
val,test=train_test_split(rem,test_size=.50,random_state=42,stratify=rem.sentiment)
m=joblib.load(root/'sentiment_model.joblib');pred=m.predict(test.content)
r=classification_report(test.sentiment,pred,output_dict=True,zero_division=0)
tp=int(((test.sentiment=='negative')&(pred=='negative')).sum());fn=int(((test.sentiment=='negative')&(pred!='negative')).sum())
out={'date':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'method':'Reconstructed existing split from baseline code; loaded saved model, no fitting or tuning','split':{'train':len(train),'validation':len(val),'test':len(test),'seed':42},'TP_negative':tp,'FN_negative':fn,'negative_support':tp+fn,'recall_negative':tp/(tp+fn),'macro_f1':r['macro avg']['f1-score'],'labels':['negative','neutral','positive'],'confusion_matrix':confusion_matrix(test.sentiment,pred,labels=['negative','neutral','positive']).tolist(),'classification_report':r,'model_sha256':hashlib.sha256((root/'sentiment_model.joblib').read_bytes()).hexdigest(),'dataset_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'test_text_keys_sha256':hashlib.sha256('\n'.join(test.text_key).encode()).hexdigest(),'python':sys.version,'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__,'limitation':'Repeated evaluation of existing test; not a new independent sample. Historical per-row split manifest absent; reconstructed from source and data.'}
destination=args.output or root/'evidence/ml_test_evidence_rerun.json'
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
