"""在 MNIST 上验证 AE/VAE，并保存原图与重建图。"""

import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from dl_models import Autoencoder, VariationalAutoencoder
from dl_models.autoencoders import vae_loss


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--model",choices=["ae","vae"],required=True)
    parser.add_argument("--epochs",type=int,default=5); parser.add_argument("--limit",type=int,default=12000); args=parser.parse_args()
    torch.manual_seed(42); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=datasets.MNIST("data",train=True,download=True,transform=transforms.ToTensor())
    test=datasets.MNIST("data",train=False,download=True,transform=transforms.ToTensor())
    if args.limit: train=Subset(train,range(min(args.limit,len(train))))
    loader=DataLoader(train,64,shuffle=True); test_loader=DataLoader(test,256)
    model=(Autoencoder(784,16) if args.model=="ae" else VariationalAutoencoder(784,16)).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    def loss_for(output,x): return nn.functional.binary_cross_entropy(output[0],x) if args.model=="ae" else vae_loss(output[0],x,output[1],output[2])
    for epoch in range(args.epochs):
        model.train(); total=0.0
        for images,_ in loader:
            x=images.to(device).flatten(1); output=model(x); loss=loss_for(output,x)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total+=loss.item()*len(x)
        print(f"epoch={epoch+1:02d} train_loss={total/len(train):.4f}")
    model.eval(); images,_=next(iter(test_loader)); x=images.to(device).flatten(1)
    with torch.inference_mode(): reconstruction=model(x)[0].reshape(-1,1,28,28)
    Path("outputs").mkdir(exist_ok=True); save_image(torch.cat([images[:16],reconstruction[:16].cpu()]),f"outputs/{args.model}_reconstruction.png",nrow=16)
    print(f"saved outputs/{args.model}_reconstruction.png")


if __name__=="__main__": main()
