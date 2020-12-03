# delete this line if you want disable fold option in vim.
# vim:set foldmethod=marker:
"""
Show raw audio and mu-law encode samples to make input source
"""
import os,sys

# import librosa # use only for raw audio input.
import numpy as np

import torch
import torch.utils.data as data

import pickle

from IPython import embed

# general audio methods {{{
def load_audio(filename, sample_rate=16000, trim=True, trim_frame_length=2048):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)

    if trim:
        audio, _ = librosa.effects.trim(audio, frame_length=trim_frame_length)

    return audio


def one_hot_encode(data, channels=256):
    one_hot = np.zeros((data.size, channels), dtype=float)
    one_hot[np.arange(data.size), data.ravel()] = 1

    return one_hot


def one_hot_decode(data, axis=1):
    decoded = np.argmax(data, axis=axis)

    return decoded


def mu_law_encode(audio, quantization_channels=256):
    """
    Quantize waveform amplitudes.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)
    quantize_space = np.linspace(-1, 1, quantization_channels)

    quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
    quantized = np.digitize(quantized, quantize_space) - 1

    return quantized


def mu_law_decode(output, quantization_channels=256):
    """
    Recovers waveform from quantized values.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)

    expanded = (output / quantization_channels) * 2. - 1
    waveform = np.sign(expanded) * (
                   np.exp(np.abs(expanded) * np.log(mu + 1)) - 1
               ) / mu

    return waveform

# }}}
# Dataset class (original) {{{
    """
    input: raw audio
    output: raw audio
    """

class Dataset(data.Dataset):
    def __init__(self, data_dir, sample_rate=16000, in_channels=256, trim=False): #no trim for match timestep
        super(Dataset, self).__init__()

        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.trim = trim

        self.root_path = data_dir
        self.filenames = [x for x in sorted(os.listdir(data_dir))]

    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filenames[index])

        raw_audio = load_audio(filepath, self.sample_rate, self.trim)

        encoded_audio = mu_law_encode(raw_audio, self.in_channels)
        encoded_audio = one_hot_encode(encoded_audio, self.in_channels)

        return encoded_audio

    def __len__(self):
        return len(self.filenames)


class DataLoader(data.DataLoader):
    def __init__(self, data_dir, receptive_fields,
                 sample_size=0, sample_rate=16000, in_channels=256,
                 batch_size=1, shuffle=True):
        """
        DataLoader for WaveNet
        :param data_dir:
        :param receptive_fields: integer. size(length) of receptive fields
        :param sample_size: integer. number of timesteps to train at once.
                            sample size has to be bigger than receptive fields.
                            |-- receptive field --|---------------------|
                            |------- samples -------------------|
                            |---------------------|-- outputs --|
        :param sample_rate: sound sampling rates
        :param in_channels: number of input channels
        :param batch_size:
        :param shuffle:
        """
        dataset = Dataset(data_dir, sample_rate, in_channels)

        super(DataLoader, self).__init__(dataset, batch_size, shuffle)

        if sample_size <= receptive_fields:
            raise Exception("sample_size has to be bigger than receptive_fields")

        self.sample_size = sample_size
        self.receptive_fields = receptive_fields

        self.collate_fn = self._collate_fn

    def calc_sample_size(self, audio):
        return self.sample_size if len(audio[0]) >= self.sample_size\
                                else len(audio[0])

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    def _collate_fn(self, audio):
        audio = np.pad(audio, [[0, 0], [self.receptive_fields, 0], [0, 0]], 'constant')

        if self.sample_size:
            # サンプルサイズが設定されている場合はその長さに音源を切って順にトレーニングに投げる(ランダムにはならない！)
            sample_size = self.calc_sample_size(audio)

            while sample_size > self.receptive_fields:
                inputs = audio[:, :sample_size, :]
                targets = audio[:, self.receptive_fields:sample_size, :]

                yield self._variable(inputs),\
                      self._variable(one_hot_decode(targets, 2))

                audio = audio[:, sample_size-self.receptive_fields:, :]
                sample_size = self.calc_sample_size(audio)
        else:
            # サンプルサイズが設定されていない場合はそのまま投げる(メモリ溢れても知らないよ)
            targets = audio[:, self.receptive_fields:, :]
            return self._variable(audio),\
                   self._variable(one_hot_decode(targets, 2))
# }}}
# Dataset class (ddc_onsetnet) {{{
    """
    input: spectrum (one channel from original data)
    output: onset prediction(0~1, 1channel)

    data_dir -> ddc's chart_onset/mel80hop441
    train.txtとかが入っているがこれは消しておく
    とりあえずで動かしたいので
        * バッチサイズ１(一曲ずつ読み込む)
        * サンプルサイズ＝一曲まるまる(音ゲーなので差はあれど長さはほとんど同じとみなす)
    """

class Dataset_onset(data.Dataset):
    def __init__(self, data_dir, selected_channel=0, in_channels=80, trim=False): #no trim for match timestep
        super(Dataset_onset, self).__init__()

        self.in_channels = in_channels
        self.trim = trim

        self.root_path = data_dir
        self.filenames = [x for x in sorted(os.listdir(self.root_path)) if 'pkl' in x]

    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filenames[index])

        with open(filepath, 'rb') as f:
            file = pickle.load(f)

        return file

    def __len__(self):
        return len(self.filenames)


class DataLoader_onset(data.DataLoader):
    def __init__(self, data_dir, receptive_fields, in_channels=80,
                 batch_size=1, shuffle=True, valid=False):
        """
        DataLoader for WaveNet
        :param data_dir:
        :param receptive_fields: integer. size(length) of receptive fields
                            sample size has to be bigger than receptive fields.
                            |-- receptive field --|---------------------|
                            |------- samples -------------------|
                            |---------------------|-- outputs --|
        :param in_channels: number of input channels
        :param batch_size:
        :param shuffle:
        """
        # add validation feature. if valid=True, this is dataloader for validation
        dataset = Dataset_onset(data_dir, in_channels)
        dataset_size = len(dataset)
        train_size = int(dataset_size*0.9)
        if valid==False:
            dataset = data.dataset.Subset(dataset, list(range(train_size)))
        else:
            dataset = data.dataset.Subset(dataset, list(range(train_size, dataset_size)))

        super(DataLoader_onset, self).__init__(dataset, batch_size, shuffle)

        # if sample_size <= receptive_fields:
        #     raise Exception("sample_size has to be bigger than receptive_fields")

        # self.sample_size = sample_size
        self.receptive_fields = receptive_fields

        self.collate_fn = self._collate_fn

    def calc_sample_size(self, audio):
        len(audio[0])

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    def _collate_fn(self, files):
        song_feat_batch = []
        chart_batch = []
        for file in files:
            # 今はbatch_size:1を想定、そうでないと曲によって長さが違うのでちょっと処理が面倒になる
            song_meta, song_feat, charts = file
            # song_feat
            song_feat = np.squeeze(song_feat[:, :, 0:1]) # select first channel for now #TODO selective channel
            song_feat_batch.append(song_feat)
            # chart
            for chart in charts:
                coarse_diff = chart.get_coarse_difficulty()
                if coarse_diff == "Challenge":
                    chart_batch.append(chart)
                    break

        target_batch = []
        for chart in chart_batch: # batch_size=1 batch次元が必要なので形式上のfor文
            target = [int(frame_idx in chart.onsets) for frame_idx in range(chart.nframes)]
            target_batch.append(target)

        song_feat_batch = np.array(song_feat_batch)
        song_feat_batch = np.pad(song_feat_batch, [[0, 0], [self.receptive_fields, 0], [0, 0]], 'constant')
        target_batch = np.array(target_batch)

        # オリジナル実装では、sample_sizeを制限すると曲を刻んでyieldするので、それに合わせてyieldになっている。
        yield self._variable(song_feat_batch), self._variable(target_batch)
# }}}
