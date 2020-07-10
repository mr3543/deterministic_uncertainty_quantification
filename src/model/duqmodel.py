import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.backbone import Backbone

class DUQ(nn.Module):
    def __init__(self,sigma,gamma,num_classes,emb_size):
        super(DUQ,self).__init__()

        self.backbone = Backbone(emb_size)
        self.num_classes = num_classes
        self.sigma = sigma
        self.gamma = gamma
        self.emb_size = emb_size


        self.register_buffer("n",torch.ones(num_classes) * 12)
        self.register_buffer(
                "m",torch.normal(torch.zeros(num_classes,emb_size),1)
        )

        self.W = nn.Parameter(
                torch.normal(torch.zeros(num_classes,emb_size,256),0.05)
        )

    
    def kernel(self,Wx):
        # Wx: [B,C,D]
        # cntrds: [C,D]
        cntrds = self.m/self.n[:,None]
        K = (-(Wx - cntrds)**2).mean(-1).div(2 * self.sigma**2).exp()
        return K
    
    def update_centroids(self,Wx,targets):
        # Wx: [B,C,D]
        # targets: [B,C]

        cls_embeddings = torch.einsum('bc,bcd -> cd',targets,Wx)
        nt = targets.sum(0)

        self.n = self.gamma*self.n + (1-self.gamma)*nt
        self.m = self.gamma*self.m + (1-self.gamma)*cls_embeddings

    def loss(self,K,targets):
        return F.binary_cross_entropy(K,targets)

    def forward(self,x):
        x  = self.backbone(x)

        # x: [B,D]
        # self.W: [C,D,E]
        Wx = torch.einsum('bd,cde -> bce',x,self.W)
        return Wx



