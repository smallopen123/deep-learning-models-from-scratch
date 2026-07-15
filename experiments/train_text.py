"""用仓库内可审阅情感数据验证 RNN/LSTM/GRU/Transformer。"""

import argparse
import csv
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dl_models import SequenceClassifier, TransformerClassifier


def load_rows():
    with Path("data/tiny_sentiment.tsv").open(encoding="utf-8") as file:
        return [(row["text"].lower().split(),int(row["label"])) for row in csv.DictReader(file,delimiter="\t")]


def encode(rows,max_length=12):
    counter=Counter(token for tokens,_ in rows for token in tokens)
    vocab={token:index+2 for index,(token,_) in enumerate(counter.most_common())}; vocab["<pad>"]=0; vocab["<unk>"]=1
    features=[]; labels=[]
    for tokens,label in rows:
        ids=[vocab.get(token,1) for token in tokens][:max_length]; ids+=([0]*(max_length-len(ids)))
        features.append(ids); labels.append(label)
    return torch.tensor(features),torch.tensor(labels),vocab


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--model",choices=["rnn","lstm","gru","transformer"],required=True); parser.add_argument("--epochs",type=int,default=30); args=parser.parse_args()
    torch.manual_seed(42); rows=load_rows(); x,y,vocab=encode(rows)
    # 固定交错划分：每类最后 4 条进入测试集。
    positive=torch.where(y==1)[0]; negative=torch.where(y==0)[0]; test_ids=torch.cat([positive[-4:],negative[-4:]]); train_ids=torch.cat([positive[:-4],negative[:-4]])
    loader=DataLoader(TensorDataset(x[train_ids],y[train_ids]),batch_size=8,shuffle=True)
    model=TransformerClassifier(len(vocab),32,4,2,2) if args.model=="transformer" else SequenceClassifier(len(vocab),16,32,2,args.model)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-3); loss_fn=nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train(); total=0.0
        for bx,by in loader:
            loss=loss_fn(model(bx),by); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); total+=loss.item()*len(bx)
        if (epoch+1)%5==0: print(f"epoch={epoch+1:02d} train_loss={total/len(train_ids):.4f}")
    model.eval()
    with torch.inference_mode(): predictions=model(x[test_ids]).argmax(1); accuracy=(predictions==y[test_ids]).float().mean()
    print("predictions=",predictions.tolist()); print("targets=",y[test_ids].tolist()); print(f"test_accuracy={accuracy.item():.3f}")


if __name__=="__main__": main()
