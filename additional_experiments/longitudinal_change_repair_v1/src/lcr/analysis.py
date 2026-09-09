from __future__ import annotations
import numpy as np

def paired_bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, draws: int = 10_000, seed: int = 20260908) -> tuple[float,float,float]:
    """Paired R²(a)-R²(b) confidence interval. Caller supplies one prediction per patient/task."""
    rng=np.random.default_rng(seed); idx=np.arange(len(y))
    def r2(p, take):
        yy=y[take]; return 1-np.square(yy-p[take]).sum()/max(np.square(yy-yy.mean()).sum(),1e-8)
    point=r2(a,idx)-r2(b,idx); values=np.array([r2(a,t:=rng.choice(idx,len(idx),replace=True))-r2(b,t) for _ in range(draws)])
    return float(point),float(np.quantile(values,.025)),float(np.quantile(values,.975))

def holm(pvalues: list[float]) -> list[float]:
    order=np.argsort(pvalues); out=np.empty(len(pvalues)); running=0.
    for rank,i in enumerate(order): running=max(running,(len(pvalues)-rank)*pvalues[i]); out[i]=min(running,1.)
    return out.tolist()
