import torch
from copy import deepcopy

class ModelEMA(object):
    """ 
    Model Exponential Moving Average V4 (核弹级修复版)
    严格区分 Parameter (权重) 和 Buffer (BN统计量)
    """
    def __init__(self, model, decay=0.9999, device=None):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def update(self, model):
        with torch.no_grad():
            # 1. 只对 "参数" (Weights/Biases) 进行 EMA 平滑
            # 使用 named_parameters 自动递归查找所有权重
            msd = dict(model.named_parameters())
            esd = dict(self.module.named_parameters())
            
            for k in msd:
                model_v = msd[k].detach()
                ema_v = esd[k]
                
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                
                # EMA 公式: new = old * decay + current * (1 - decay)
                ema_v.mul_(self.decay).add_(model_v, alpha=1. - self.decay)

            # 2. 对 "缓冲区" (BN Stats: running_mean, var) 进行直接覆盖
            # 绝对不要平均！直接 Copy！
            msd_buf = dict(model.named_buffers())
            esd_buf = dict(self.module.named_buffers())
            
            for k in msd_buf:
                model_v = msd_buf[k].detach()
                ema_v = esd_buf[k]
                
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                
                # 硬拷贝
                ema_v.copy_(model_v)