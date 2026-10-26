import argparse, csv, json
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--csv',required=True);p.add_argument('--labels',required=True,help='JSON mapping model label to dataset class');a=p.parse_args()
    mapping=json.loads(a.labels)
    with open(a.csv,newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError('Empty evaluation dataset')
    classifier=pipeline('text-classification',model=a.model,tokenizer=a.model)
    results=classifier([r['tweet'] for r in rows],truncation=True,batch_size=8)
    truth=[r['class'] for r in rows];pred=[str(mapping[r['label']]) for r in results]
    print(json.dumps({'n':len(rows),'accuracy':accuracy_score(truth,pred),'macro_f1':f1_score(truth,pred,average='macro'),'confusion_matrix':confusion_matrix(truth,pred).tolist()},indent=2))
