import torch
import torch.nn as nn
import numpy as np

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, max_iter):
        ctx.alpha = alpha
        ctx.max_iter = max_iter
        ctx.iter_num = ctx.iter_num + 1 if hasattr(ctx, 'iter_num') else 1
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.alpha
        max_iter = ctx.max_iter
        iter_num = ctx.iter_num
        coeff = np.float32(2.0 * (1.0 - 0.0) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - 1.0 + 0.0)
        return -coeff * grad_output, None, None

class Discriminator(nn.Module):
    def __init__(self, channels=1024, num_classes=4, alpha=10, max_iter=4000):
        super(Discriminator, self).__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, num_classes)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )

    def forward(self, x):
        adversarial_out = self.ad_net(GRL.apply(x, self.alpha, self.max_iter))
        return adversarial_out