from __future__ import annotations
import copy, math
import torch
from torch import nn
from .contracts import SUMMARY_SHAPE, SPATIAL_SHAPE

def sinusoid(n,d):
    p=torch.arange(n).float()[:,None]; q=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000)/d)); z=torch.zeros(n,d); z[:,0::2]=torch.sin(p*q); z[:,1::2]=torch.cos(p*q); return z

class GlobalAdapter(nn.Module):
    def __init__(self,dropout=0.):
        super().__init__(); self.summary_projection=nn.Sequential(nn.Linear(2304,128),nn.LayerNorm(128),nn.GELU()); self.channel_embedding=nn.Parameter(torch.randn(1,7,128)/math.sqrt(128)); self.channel_transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,256,dropout=dropout,activation="gelu",batch_first=True,norm_first=True),1); self.state_token=nn.Parameter(torch.randn(1,1,128)/math.sqrt(128)); self.register_buffer("axial_position",sinusoid(33,128)[None]); self.slice_transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,512,dropout=dropout,activation="gelu",batch_first=True,norm_first=True),1); self.response_projection=nn.Sequential(nn.Linear(128,192),nn.LayerNorm(192))
    def forward(self,x):
        if x.ndim!=5 or tuple(x.shape[1:])!=SUMMARY_SHAPE or not torch.isfinite(x).all():raise ValueError("global summary contract failed")
        b,v,c,s,_=x.shape; z=self.summary_projection(x.to(self.summary_projection[0].weight.dtype)); z=z.permute(0,1,3,2,4).reshape(-1,c,128); z=self.channel_transformer(z+self.channel_embedding).mean(1).reshape(b*v,s,128); z=torch.cat((self.state_token.expand(b*v,-1,-1),z),1)+self.axial_position; return self.response_projection(self.slice_transformer(z)[:,0]).reshape(b,v,192)

class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.proj=nn.Sequential(nn.Linear(768,128),nn.LayerNorm(128),nn.GELU()); self.conv=nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.GELU(),nn.Conv2d(128,128,3,padding=1)); self.norm=nn.LayerNorm(128); self.pos=nn.Parameter(sinusoid(49,128).reshape(1,7,7,128).permute(0,3,1,2)); self.score=nn.Conv2d(128,1,1)
    def forward(self,x):
        b,v,c,s,hy,hx,d=x.shape; z=self.proj(x.reshape(-1,hy,hx,d)).permute(0,3,1,2); z=z+self.pos; z=z+self.conv(z); weights=torch.softmax(self.score(z).flatten(1),1); pooled=(z.flatten(2)*weights[:,None]).sum(-1); return self.norm(pooled).reshape(b,v,c,s,128)

class Branch64(nn.Module):
    def __init__(self):super().__init__(); self.net=nn.Sequential(nn.Linear(192,64),nn.LayerNorm(64))
    def forward(self,x):return self.net(x)

class ParallelSpatialModel(nn.Module):
    def __init__(self,c0_checkpoint: str,arm: str):
        super().__init__(); payload=torch.load(c0_checkpoint,map_location="cpu",weights_only=False); src=payload["model_state"]; self.arm=arm; self.c0=GlobalAdapter(0.); self.c0.load_state_dict({k[8:]:v for k,v in src.items() if k.startswith("adapter.")},strict=True); self.c0.requires_grad_(False); self.c0.eval(); self.rad_adapter=copy.deepcopy(self.c0); self.rad_branch=Branch64(); self.rad_branch.load_state_dict({k[17:]:v for k,v in src.items() if k.startswith("phenotype_branch.")},strict=True); self.spatial=SpatialEncoder() if arm=="S1_SPATIAL" else None; self.spatial_to_adapter=nn.Linear(128,128) if self.spatial is not None else None
        if self.spatial_to_adapter is not None: nn.init.zeros_(self.spatial_to_adapter.weight); nn.init.zeros_(self.spatial_to_adapter.bias)
        self.heads=nn.ModuleList([nn.Linear(64,4) for _ in range(4)])
    def train(self,mode=True):super().train(mode); self.c0.eval(); return self
    def forward(self,summary,spatial_tokens=None):
        c0=self.c0(summary).detach(); g=self.rad_adapter.summary_projection(summary.to(self.rad_adapter.summary_projection[0].weight.dtype)); b,v,c,s,_=g.shape
        if self.spatial is not None:
            if spatial_tokens is None or tuple(spatial_tokens.shape[1:])!=SPATIAL_SHAPE:raise ValueError("spatial token contract failed")
            local=self.spatial(spatial_tokens.to(g.dtype)); add=self.spatial_to_adapter(local); g=g.permute(0,1,3,2,4).reshape(-1,c,128); add=add.permute(0,1,3,2,4).reshape(-1,c,128); g=(g+add).reshape(b,v,s,c,128)
        else:g=g.permute(0,1,3,2,4)
        g=self.rad_adapter.summary_projection[1:](g); g=self.rad_adapter.channel_transformer(g.reshape(-1,c,128)+self.rad_adapter.channel_embedding).mean(1).reshape(b*v,s,128); z=torch.cat((self.rad_adapter.state_token.expand(b*v,-1,-1),g),1)+self.rad_adapter.axial_position; z=self.rad_adapter.response_projection(self.rad_adapter.slice_transformer(z)[:,0]).reshape(b,v,192); rad=self.rad_branch(z); pred=torch.cat([h(rad) for h in self.heads],-1); return {"c0_state":c0,"rad_state":rad,"prediction":pred}
    def rad_parameters(self):return [p for n,p in self.named_parameters() if not n.startswith("c0.") and p.requires_grad]
    def freeze_rad(self):self.rad_adapter.requires_grad_(False); self.rad_branch.requires_grad_(False); self.heads.requires_grad_(True); self.spatial.requires_grad_(False) if self.spatial is not None else None; self.spatial_to_adapter.requires_grad_(False) if self.spatial_to_adapter is not None else None
    def unfreeze_rad(self):self.c0.requires_grad_(False); self.rad_adapter.requires_grad_(True); self.rad_branch.requires_grad_(True); self.heads.requires_grad_(True); self.spatial.requires_grad_(True) if self.spatial is not None else None; self.spatial_to_adapter.requires_grad_(True) if self.spatial_to_adapter is not None else None
