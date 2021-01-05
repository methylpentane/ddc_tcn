import torch
import warnings

class LossForOneshot(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.loss_func_symbol = torch.nn.CrossEntropyLoss()
        self.loss_func_onset = torch.nn.BCEWithLogitsLoss()
        self.out_channels = out_channels

    def forward(self, outputs, targets):
        '''
        outputs: [1, time, (1+256)]
        targets: [1, time, (1+1)]
        '''
        onset_outputs = outputs[:,:,0:1] #[1,time,1]
        symbol_outputs = outputs[:,:,1:] #[1,time,self.out_channels]
        onset_targets = targets[:,:,0:1] #[1,time,1]
        symbol_targets = targets[:,:,1:] #[1,time,1]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            onset_index_list = onset_targets.nonzero()[:,1]

        if onset_index_list.shape[0] != 0:
            symbol_targets = symbol_targets[:, onset_index_list, :]
            symbol_outputs = symbol_outputs[:, onset_index_list, :]
            symbol_loss = self.loss_func_symbol(symbol_outputs.view(-1, self.out_channels-1), symbol_targets.view(-1).long())
        else:
            symbol_loss = torch.zeros(1).cuda()
        onset_loss = self.loss_func_onset(onset_outputs.view(-1), onset_targets.view(-1))
        loss = onset_loss + symbol_loss

        return loss
