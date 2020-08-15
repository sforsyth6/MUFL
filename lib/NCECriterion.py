import torch
from torch import nn

class NCECriterion(nn.Module):

    def __init__(self):
        super(NCECriterion, self).__init__()

    def forward(self, x, targets):
        batchSize = x.size(0)

        lnPmt = x.clone()
        lnPmt.log_()
        lnPmtsum = lnPmt.sum(0)

        loss = - (lnPmtsum) / batchSize
        return loss