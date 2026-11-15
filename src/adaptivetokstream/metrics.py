"""Reference metrics on clamped RGB [0,1]; no synthetic quality fallback."""
import math
import numpy as np


class QualityMetrics:
    def __init__(self,device):
        import torch
        import lpips
        from DISTS_pytorch import DISTS
        self.torch=torch
        self.device=device
        self.lpips=lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
        self.dists=DISTS().to(device).eval().requires_grad_(False)

    def evaluate(self,reference,reconstruction):
        torch=self.torch
        def batch(x):
            if isinstance(x,np.ndarray):
                x=torch.from_numpy(x.copy()).permute(0,3,1,2).float()/255
            elif x.ndim==5:
                x=x[0].permute(1,0,2,3)
            return x.to(self.device).float().clamp(0,1)
        a,b=batch(reference),batch(reconstruction)
        if a.shape!=b.shape:raise ValueError('Reference and reconstruction shapes differ')
        with torch.inference_mode():
            mse=torch.mean((a-b)**2).item()
            perceptual=self.lpips(a*2-1,b*2-1).mean().item()
            structural=self.dists(a,b).mean().item()
        return {'mse':mse,'psnr':-10*math.log10(max(mse,1e-12)),
                'lpips':perceptual,'dists':structural}
