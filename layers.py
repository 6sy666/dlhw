import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ZINBLoss(nn.Module):
    def __init__(self):
        # 继承 nn.Module 类
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        # eps 是一个小的值，用于保证数值稳定性
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        # 将 mean 按比例缩放
        mean = mean * scale_factor
        # 计算负二项分布损失的部分
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2
        # 计算零注入负二项分布损失的部分
        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        # 结合零注入负二项分布和负二项分布的情况
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        # 如果指定了 L2 正则化项，则添加
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        # 对所有损失值取平均值
        result = torch.mean(result)
        return result


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        # 继承 nn.Module 类
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        # 在训练模式下添加高斯噪声
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        # 将 x 取对数，并将值限制在 [1e-5, 1e6] 之间
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        # 将 x 取 softplus，并将值限制在 [1e-4, 1e4] 之间
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
