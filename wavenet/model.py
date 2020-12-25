# delete this line if you want disable fold option in vim.
# vim:set foldmethod=marker:
"""
Main model of WaveNet
Calculate loss and optimizing
"""
import os

import torch
import torch.optim
import numpy as np

from wavenet.networks import WaveNetModule
from wavenet.utils.util import find_pred_onsets, align_onsets_to_sklearn
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score

# original WaveNet {{{
class WaveNet:
    def __init__(self, layer_size, stack_size, in_channels, res_channels, out_channels, gc_channels, input_scale, lr=0.002):

        self.net = WaveNetModule(layer_size, stack_size, in_channels, res_channels, out_channels, gc_channels, input_scale)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gc_channels = gc_channels
        self.receptive_fields = self.net.receptive_fields

        self.lr = lr
        self.loss = self._loss()
        self.optimizer = self._optimizer()

        self._prepare_for_gpu()

    def _loss(self):
        if self.out_channels==1:
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def _prepare_for_gpu(self):
        if torch.cuda.device_count() > 1:
            print("{0} GPUs are detected.".format(torch.cuda.device_count()))
            self.net = torch.nn.DataParallel(self.net)

        if torch.cuda.is_available():
            self.net.cuda()

    def train(self, inputs, targets, gc_inputs=None):
        """
        Train 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :param targets: Torch tensor [batch, timestep, channels]
        :param gc_inputs: Tensor[batch, channels]
        :return: float loss
        """
        if not self.net.training:
            self.net.train()

        if gc_inputs is not None:
            assert gc_inputs.shape[1] == self.gc_channels
        else:
            assert self.gc_channels == 0

        outputs = self.net(inputs, gc_inputs)

        if self.out_channels == 1:
            loss = self.loss(outputs.view(1,-1), targets)
        else:
            loss = self.loss(outpus.view(-1,self.in_channels).long(), targets.view(1,-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def generate(self, inputs, gc_inputs=None):
        """
        Generate 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :param gc_inputs: Tensor[batch, channels]
        :return: Tensor[batch, timestep, channels]
        """
        if self.net.training:
            self.net.eval()

        if gc_inputs is not None:
            assert gc_inputs.shape[1] == self.gc_channels
        else:
            assert self.gc_channels == 0

        with torch.no_grad():
            outputs = self.net(inputs, gc_inputs)

        return outputs

    @staticmethod
    def get_model_path(model_dir, step=0):
        basename = 'wavenet'

        if step:
            return os.path.join(model_dir, '{0}_{1}.pkl'.format(basename, step))
        else:
            return os.path.join(model_dir, '{0}.pkl'.format(basename))

    def load(self, model_dir, step=0):
        """
        Load pre-trained model
        :param model_dir:
        :param step:
        :return:
        """
        print("Loading model from {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        self.net.load_state_dict(torch.load(model_path))

    def save(self, model_dir, step=0):
        print("Saving model into {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        torch.save(self.net.state_dict(), model_path)

# }}}
# ddc wavenet {{{
"""
モデルは共通ではあるものの、validationにおいて最終的なonsetはしきい値処理という後処理工程を挟むので別クラス
"""
class WaveNet_onset(WaveNet):
    def __init__(self, layer_size, stack_size, in_channels, res_channels, out_channels, gc_channels, input_scale, lr=0.002):
        super(WaveNet_onset, self).__init__(layer_size, stack_size, in_channels, res_channels, out_channels, gc_channels, input_scale, lr)

    def validation(self, inputs, targets, gc_inputs=None):
        """
        validation 1 time
        :param inputs: Tensor[batch=1, timestep, channels=1]
        :param targets: tensor [batch=1, timestep]
        :param gc_inputs: Tensor[batch, channels]
        :return: metrics for 1 generation
        """
        if gc_inputs is not None:
            assert gc_inputs.shape[1] == self.gc_channels
        else:
            assert self.gc_channels == 0
        result = self.generate(inputs, gc_inputs)

        # preprocess
        result = np.squeeze(result.to('cpu').detach().numpy().copy())
        targets = np.squeeze(targets.to('cpu').detach().numpy().copy())
        assert result.shape == targets.shape

        window = np.hamming(5)
        predicted_onsets = find_pred_onsets(result, window)
        true_onsets = set(np.where(targets == 1)[0].tolist())
        y_true, y_scores_pkalgn = align_onsets_to_sklearn(true_onsets, predicted_onsets, result, tolerance=2)

        # eval_metrics_for_scores(ddc onset_train.py)
        nonsets = np.sum(y_true)
        # calculate ROC curve
        fprs, tprs, thresholds = roc_curve(y_true, y_scores_pkalgn)
        auroc = auc(fprs, tprs)

        # calculate PR curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores_pkalgn)
        # https://github.com/scikit-learn/scikit-learn/issues/1423
        auprc = auc(recalls, precisions)

        # find best fscore and associated values
        fscores_denom = precisions + recalls
        fscores_denom[np.where(fscores_denom == 0.0)] = 1.0
        fscores = (2 * (precisions * recalls)) / fscores_denom
        fscore_max_idx = np.argmax(fscores)
        precision, recall, fscore, threshold_ideal = precisions[fscore_max_idx], recalls[fscore_max_idx], fscores[fscore_max_idx], thresholds[fscore_max_idx]

        # calculate density
        predicted_steps = np.where(y_scores_pkalgn >= threshold_ideal)
        density_rel = float(len(predicted_steps[0])) / float(nonsets)

        # calculate accuracy
        y_labels = np.zeros(y_scores_pkalgn.shape[0], dtype=np.int)
        y_labels[predicted_steps] = 1
        accuracy = accuracy_score(y_true.astype(np.int), y_labels)

        # aggregate metrics
        metrics = {}
        metrics['auroc'] = auroc
        metrics['auprc'] = auprc
        metrics['fscore'] = fscore
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['threshold'] = threshold_ideal
        metrics['accuracy'] = accuracy
        metrics['density_rel'] = density_rel

        # from IPython import embed
        # embed()
        # exit()

        return metrics

# }}}
