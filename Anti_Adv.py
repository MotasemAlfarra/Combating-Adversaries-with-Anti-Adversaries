import torch
import torch.nn as nn
import torch.nn.functional as F

class anti_adversary_wrapper(nn.Module):
    def __init__(self, model, mean=None, std=None, k=0, alpha=1):
        super(anti_adversary_wrapper, self).__init__()
        self.model = model
        self.normalize = False
        self.k = k
        self.alpha = alpha
        # self.loss = nn.CrossEntropyLoss()
        if mean is not None:
            assert std is not None
            self.normalize = True
            # std
            std = torch.tensor(std).view(1, 3, 1, 1)
            self.std = nn.Parameter(std, requires_grad=False).to("cuda")
            # mean
            mean = torch.tensor(mean).view(1, 3, 1, 1)
            self.mean = nn.Parameter(mean, requires_grad=False).to("cuda")
    
    def get_anti_adversary(self, x):
        sudo_label = self.model(x).max(1)[1]
        with torch.enable_grad():#because usually people disables gradients in evaluations
            anti_adv = torch.zeros_like(x, requires_grad=True)
            for _ in range(self.k):
                loss = F.cross_entropy(self.model(x+anti_adv), sudo_label)
                grad = torch.autograd.grad(loss, anti_adv)
                anti_adv.data -= self.alpha*grad[0].sign()
        return anti_adv
    
    def forward(self, x):#Adaptive update of the anti adversary
        if self.normalize:
            x = (x - self.mean) / self.std
        if self.k > 0 and self.alpha > 0:
            anti_adv = self.get_anti_adversary(x)
            return self.model(x+anti_adv)
        return self.model(x)
