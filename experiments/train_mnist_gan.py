"""在 MNIST 上验证全连接 GAN，每轮保存生成图。"""

import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from dl_models import Discriminator, Generator


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--epochs",type=int,default=10); parser.add_argument("--limit",type=int,default=20000); args=parser.parse_args()
    torch.manual_seed(42); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); noise_dim=64
    data=datasets.MNIST("data",train=True,download=True,transform=transforms.ToTensor())
    if args.limit: data=Subset(data,range(min(args.limit,len(data))))
    loader=DataLoader(data,64,shuffle=True); g=Generator(noise_dim,784).to(device); d=Discriminator(784).to(device)
    go=torch.optim.Adam(g.parameters(),2e-4,betas=(0.5,0.999)); do=torch.optim.Adam(d.parameters(),2e-4,betas=(0.5,0.999)); bce=nn.BCEWithLogitsLoss()
    fixed=torch.randn(64,noise_dim,device=device); Path("outputs").mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        for images,_ in loader:
            real=images.to(device).flatten(1); batch=len(real); fake=torch.sigmoid(g(torch.randn(batch,noise_dim,device=device)))
            d_loss=bce(d(real),torch.ones(batch,1,device=device))+bce(d(fake.detach()),torch.zeros(batch,1,device=device))
            do.zero_grad(); d_loss.backward(); do.step()
            generated=torch.sigmoid(g(torch.randn(batch,noise_dim,device=device))); g_loss=bce(d(generated),torch.ones(batch,1,device=device))
            go.zero_grad(); g_loss.backward(); go.step()
        with torch.inference_mode(): samples=torch.sigmoid(g(fixed)).reshape(-1,1,28,28)
        save_image(samples,f"outputs/gan_epoch_{epoch+1:02d}.png",nrow=8)
        print(f"epoch={epoch+1:02d} d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f}")


if __name__=="__main__": main()
