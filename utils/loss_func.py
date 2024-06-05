import torch
import torch.nn as nn
# Same as OViTANet loss_func.py

class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, e):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), e=e.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


def nll_loss(h, y, e, alpha, eps=1e-7, reduction='mean'):

    
    y = y.type(torch.int64)
    e = e.type(torch.int64)

    S = torch.cumprod(1 - h, dim=1)
    
    S_padded = torch.cat([torch.ones_like(e), S], 1)
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(h, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    uncensored_loss = -e * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = -(1-e) * torch.log(s_this)
    
    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))
    
    return loss


class CoxSurvLoss(nn.Module):
    def __init__(self, device, eps=1e-8):
        """
        This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        """
        super(CoxSurvLoss, self).__init__()
        self.device = device
        self.eps = eps

    def __call__(self, risk, t, e):
        n = len(t)
        R_mat = torch.zeros((n, n), dtype=int, device=self.device)

        # Creating the risk set matrix
        for i in range(n):
            for j in range(n):
                R_mat[i, j] = t[j] >= t[i]

        theta = risk.reshape(-1)
        exp_theta = torch.exp(theta)
        log_sum = torch.log(torch.sum(exp_theta*R_mat, dim=1) + self.eps)
        loss_cox = -torch.mean((theta - log_sum) * e)
        return loss_cox


