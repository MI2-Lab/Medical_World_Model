import torch

def family_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
    if target.shape[-1]!=16 or mask.shape[-1]!=4 or mask[:,3].any(): raise ValueError("T3 target contract failed")
    losses=[]; visits=mask[:,:3].sum()
    for family in range(4):
        keep=mask[:,:3]
        if not bool(keep.any()): continue
        value=torch.nn.functional.smooth_l1_loss(prediction[:,:3,family*4:(family+1)*4],target[:,:3,family*4:(family+1)*4],reduction="none")
        losses.append(value[keep.unsqueeze(-1).expand_as(value)].mean())
    if not losses: raise RuntimeError("empty radiomics target batch")
    return torch.stack(losses).mean(), visits
