#!/usr/bin/env python3
"""Train/evaluate a TF-IDF + Linear SVM document classifier on the synthetic corpus."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset",default="dataset/synthetic")
    ap.add_argument("--output",default="results/ml")
    args=ap.parse_args()
    root=Path(args.dataset); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((root/"manifest.csv").open(encoding="utf-8")))
    data=[]
    for row in rows:
        text=(root/row["transcript_path"]).read_text(encoding="utf-8")
        data.append({**row,"text":text})
    train=[r for r in data if r["split"]=="train"]
    test=[r for r in data if r["split"]=="test"]
    model=Pipeline([
        ("tfidf",TfidfVectorizer(ngram_range=(1,2),lowercase=True,sublinear_tf=True)),
        ("classifier",LinearSVC(class_weight="balanced",random_state=42))
    ])
    model.fit([r["text"] for r in train],[r["document_type"] for r in train])
    pred=model.predict([r["text"] for r in test])
    labels=sorted({r["document_type"] for r in data})
    metrics={
        "model":"TF-IDF word/bigram features + LinearSVC",
        "dataset":"synthetic academic demonstration corpus",
        "train_documents":len(train),"test_documents":len(test),
        "accuracy":round(float(accuracy_score([r["document_type"] for r in test],pred)),4),
        "classification_report":classification_report([r["document_type"] for r in test],pred,labels=labels,output_dict=True,zero_division=0),
        "labels":labels,
        "confusion_matrix":confusion_matrix([r["document_type"] for r in test],pred,labels=labels).tolist(),
        "warning":"Synthetic dataset metrics are a pipeline demonstration, not evidence of real-world insurance performance."
    }
    joblib.dump(model,out/"document_classifier.joblib")
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    with (out/"test_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["document_id","actual","predicted"])
        for r,p in zip(test,pred): w.writerow([r["document_id"],r["document_type"],p])
    print(json.dumps({"accuracy":metrics["accuracy"],"train":len(train),"test":len(test),"model":metrics["model"]},indent=2))
    print("Saved model and metrics to",out)
if __name__=="__main__": main()
