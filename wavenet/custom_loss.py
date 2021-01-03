from torch import nn

class LossForOneshot(nn.Module):
    def __init__(self):
        super().__init__()
        loss_symbol = nn.CrossEntropyLoss()
        loss_onsets = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        pass
