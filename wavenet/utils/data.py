# delete this line if you want disable fold option in vim.
# vim:set foldmethod=marker:
"""
Show raw audio and mu-law encode samples to make input source
"""
import os,sys

import librosa, warnings
import numpy as np

import torch
import torch.utils.data as data

import pickle
import random

from wavenet.utils.util import load_id_dict

# general audio methods {{{
def load_audio(filename, sample_rate=16000, trim=True, trim_frame_length=2048):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
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

# Dataset class (onset_spectre) {{{
    """
    input: spectrum (one channel from original data)
    output: onset prediction(0~1, 1channel)

    data_dir -> ddc's chart_onset/mel80hop441
    chartファイル生成後に入ってるtrain.txtとかは消しておく
    曲の長さがだいたい同じだけどまちまちだから、
        * バッチサイズ１(一曲ずつ読み込む)
        * サンプルサイズ＝一曲まるまる
    """

class Dataset_onset(data.Dataset):
    def __init__(self, data_dir, trim=False): #no trim for match timestep
        super(Dataset_onset, self).__init__()

        self.trim = trim

        self.root_path = data_dir
        self.filenames = [x for x in sorted(os.listdir(self.root_path)) if x.split('.')[-1] == 'pkl']

    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filenames[index])

        with open(filepath, 'rb') as f:
            file = pickle.load(f)

        return file

    def __len__(self):
        return len(self.filenames)


class DataLoader_onset(data.DataLoader):
    def __init__(self, data_dir, receptive_fields, ddc_channel_select,
                 batch_size=1, shuffle=True, valid=False):
        """
        DataLoader for WaveNet
        :param data_dir:
        :param receptive_fields: integer. size(length) of receptive fields
                            sample size has to be bigger than receptive fields.
                            |-- receptive field --|---------------------|
                            |------- samples -------------------|
                            |---------------------|-- outputs --|
        :param ddc_channel_select: select channel of ddc input melspectrogram.
        :param batch_size:
        :param shuffle:
        """
        # add validation feature. if valid=True, this is dataloader for validation
        datasets = []
        for d_dir in data_dir:
            dataset = Dataset_onset(d_dir)
            dataset_size = len(dataset)
            train_size = int(dataset_size*0.9)
            if valid==False:
                dataset = data.Subset(dataset, list(range(train_size)))
            else:
                dataset = data.Subset(dataset, list(range(train_size, dataset_size)))
            datasets.append(dataset)
        dataset = data.ConcatDataset(datasets)
        super(DataLoader_onset, self).__init__(dataset, batch_size, shuffle)

        # if sample_size <= receptive_fields:
        #     raise Exception("sample_size has to be bigger than receptive_fields")

        # self.sample_size = sample_size
        self.receptive_fields = receptive_fields
        self.ddc_channel_select = ddc_channel_select
        self.diffs = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']

        self.collate_fn = self._collate_fn

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    def _collate_fn(self, files):
        file_chart = files[0]
        _, song_feat, charts = file_chart

        # song_feat
        song_feat = song_feat[:, :, self.ddc_channel_select] # slice selected channel
        song_feat = song_feat.reshape(-1, song_feat.shape[1]*song_feat.shape[2]) # [time, freq*channel]
        song_feat_batch = np.array([song_feat]) # batch_size:1
        song_feat_batch = np.pad(song_feat_batch, [[0, 0], [self.receptive_fields, 0], [0, 0]], 'constant')

        # chart
        target_batch_iter = []
        diff_batch_iter = []
        # データ水増しのために、譜面の左右反転などが行われているが、これはonsetデータとしては意味をなさないので、各難易度について一回しか学習を投げない
        diff_seen_flag = {key:False for key in self.diffs}
        for chart in random.sample(charts, len(charts)):
            diff = chart.get_coarse_difficulty()
            if diff_seen_flag[diff] == False:
                diff_seen_flag[diff] = True
                target = [int(frame_idx in chart.onsets) for frame_idx in range(chart.nframes)]
                target_batch = np.array([target])
                target_batch_iter.append(target_batch)
                diff_batch = np.zeros((1,5))
                diff_batch[0][self.diffs.index(diff)] = 1.0
                diff_batch_iter.append(diff_batch)

        # target_batch_iterというリストの中に一曲のchartをすべて入れてイテレーションをするので、yieldを利用する
        # unrollingと名付けたリスト(len=1)でバッチを投げているのは、切り抜いてサンプルサイズ毎に学習する場合に合わせているから
        for target_batch, diff_batch in zip(target_batch_iter, diff_batch_iter):
            song_feat_batch_unrolling = [self._variable(song_feat_batch)]
            target_batch_unrolling = [self._variable(target_batch)]
            diff_batch_unrolling = [self._variable(diff_batch)]
            yield (song_feat_batch_unrolling, diff_batch_unrolling), target_batch_unrolling
# }}}
# Dataset class (onset_raw){{{
    """
    input: raw_audio
    output: onset prediction(0~1, 1channel)

    data_dir -> ddc's chart_onset/mel80hop441
    chartファイル生成後に入ってるtrain.txtとかは消しておく

    * このフォルダの中に、生音源.oggもまとめて入れて、拡張子で場合分けしたファイルリストを作る(これならspectreと競合しない)
    * 16000Hzのまま読み込むので、パラメータによってsample_sizeによる切り分けを必要とする、その際frame_rateの兼ね合いで、指定したsample_sizeでピッタリにはならない
    * また、sample_sizeはさらに入力値+self.receptive_fieldsとする。オリジナルでは、sample_sizeがすでにreceptive_fieldsを含んでいるとする
    """

class Dataset_onset_raw(data.Dataset):
    def __init__(self, data_dir, sample_rate=16000, in_channels=256, trim=False): #no trim for match timestep
        super(Dataset_onset_raw, self).__init__()

        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.trim = trim

        self.root_path = data_dir
        self.filenames_chart = [x for x in sorted(os.listdir(self.root_path)) if x.split('.')[-1] == 'pkl']
        self.filenames_audio = [x for x in sorted(os.listdir(self.root_path)) if x.split('.')[-1] == 'ogg']

    def __getitem__(self, index):
        filepath_chart = os.path.join(self.root_path, self.filenames_chart[index])
        filepath_audio = os.path.join(self.root_path, self.filenames_audio[index])

        with open(filepath_chart, 'rb') as f:
            file_chart = pickle.load(f)

        raw_audio = load_audio(filepath_audio, self.sample_rate, self.trim)
        encoded_audio = mu_law_encode(raw_audio, self.in_channels)
        encoded_audio = one_hot_encode(encoded_audio, self.in_channels)

        item = (file_chart, encoded_audio)
        return item

    def __len__(self):
        return len(self.filenames_chart)


class DataLoader_onset_raw(data.DataLoader):
    def __init__(self, data_dir, receptive_fields,
                 sample_size=0, sample_rate=16000, chart_sample_rate=100, in_channels=256,
                 batch_size=1, shuffle=True, valid=False):
        """
        DataLoader for WaveNet
        :param data_dir:
        :param receptive_fields: integer. size(length) of receptive fields
                            sample size has to be bigger than receptive fields.
                            |-- receptive field --|---------------------|
                            |------- samples -------------------|
                            |---------------------|-- outputs --|
        :param ddc_channel_select: select channel of ddc input melspectrogram.
        :param in_channels: number of input channels
        :param batch_size:
        :param shuffle:
        """
        # add validation feature. if valid=True, this is dataloader for validation
        datasets = []
        for d_dir in data_dir:
            dataset = Dataset_onset_raw(d_dir)
            dataset_size = len(dataset)
            train_size = int(dataset_size*0.9)
            if valid==False:
                dataset = data.Subset(dataset, list(range(train_size)))
            else:
                dataset = data.Subset(dataset, list(range(train_size, dataset_size)))
            datasets.append(dataset)
        dataset = data.ConcatDataset(datasets)
        super(DataLoader_onset_raw, self).__init__(dataset, batch_size, shuffle)

        # if sample_size <= receptive_fields:
        #     raise Exception("sample_size has to be bigger than receptive_fields")
        if sample_rate % chart_sample_rate:
            raise Exception("sample_rate は 100の倍数にしてほしい")

        tmp = sample_size % (sample_rate / chart_sample_rate)
        if tmp != 0:
            self.sample_size = int(sample_size + (sample_rate / chart_sample_rate) - tmp)
            print("{}({}):sample_sizeがちょっとキリ悪いので増やしました。{}->{}".format(self.__class__.__name__,
                                                                                        ('valid' if valid else 'train'),
                                                                                        sample_size,
                                                                                        self.sample_size))
        else:
            self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.chart_sample_rate = chart_sample_rate

        self.receptive_fields = receptive_fields
        self.diffs = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']

        self.collate_fn = self._collate_fn

    def calc_sample_size(self, audio):
        if len(audio[0]) >= self.sample_size + self.receptive_fields:
            audio_sample_size = self.sample_size
            chart_sample_size = audio_sample_size / (self.sample_rate / self.chart_sample_rate)
        else:
            audio_sample_size = len(audio[0])
            chart_sample_size = (audio_sample_size - self.receptive_fields) / (self.sample_rate / self.chart_sample_rate)
            chart_sample_size = 0 if chart_sample_size < 0 else chart_sample_size
        assert chart_sample_size == int(chart_sample_size)
        return audio_sample_size, int(chart_sample_size)

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    def _collate_fn(self, files):
        file = files[0]
        file_chart, encoded_audio = file
        _, _, charts = file_chart

        # song_feat
        padding = charts[0].nframes*(self.sample_rate/self.chart_sample_rate) - encoded_audio.shape[0]
        assert padding >= 0
        song_feat_batch = np.array([encoded_audio]) # batch_size=1
        song_feat_batch = np.pad(song_feat_batch, [[0,0],[self.receptive_fields,int(padding)],[0,0]], 'constant')

        # chart
        target_batch_iter = []
        diff_batch_iter = []
        # データ水増しのために、譜面の左右反転などが行われているが、これはonsetデータとしては意味をなさないので、各難易度について一回しか学習を投げない
        diff_seen_flag = {key:False for key in self.diffs}
        for chart in random.sample(charts, len(charts)):
            diff = chart.get_coarse_difficulty()
            if diff_seen_flag[diff] == False:
                diff_seen_flag[diff] = True
                target = [int(frame_idx in chart.onsets) for frame_idx in range(chart.nframes)]
                target_batch = np.array([target])
                target_batch_iter.append(target_batch)
                diff_batch = np.zeros((1,5))
                diff_batch[0][self.diffs.index(diff)] = 1.0
                diff_batch_iter.append(diff_batch)

        # target_batch_iterというリストの中に一曲のchartをすべて入れてイテレーションをするので、yieldを利用する
        # raw_audioモデルではsample_sizeを制限すると曲を刻んでunrollingにする
        for target_batch, diff_batch in zip(target_batch_iter, diff_batch_iter):
            if self.sample_size:
                # サンプルサイズが設定されている場合はその長さに音源を切って順にトレーニングに投げる
                song_feat_batch_copy = song_feat_batch.copy()
                song_feat_batch_unrolling = []
                target_batch_unrolling = []
                diff_batch_unrolling = []
                audio_sample_size, chart_sample_size = self.calc_sample_size(song_feat_batch_copy)

                while audio_sample_size > self.receptive_fields:
                    inputs = song_feat_batch_copy[:, :audio_sample_size+self.receptive_fields, :]
                    targets = target_batch[:, :chart_sample_size]
                    song_feat_batch_unrolling.append(self._variable(inputs))
                    target_batch_unrolling.append(self._variable(targets))
                    diff_batch_unrolling.append(self._variable(diff_batch))

                    song_feat_batch_copy = song_feat_batch_copy[:, audio_sample_size:, :]
                    target_batch = target_batch[:, chart_sample_size:]
                    audio_sample_size, chart_sample_size = self.calc_sample_size(song_feat_batch_copy)

                yield (song_feat_batch_unrolling, diff_batch_unrolling), target_batch_unrolling

            else:
                # サンプルサイズが設定されていない場合はそのまま投げる(メモリ溢れても知らないよ)
                song_feat_batch_unrolling = [self._variable(song_feat_batch)]
                target_batch_unrolling = [self._variable(target_batch)]
                diff_batch_unrolling = [self._variable(diff_batch)]
                yield (song_feat_batch_unrolling, diff_batch_unrolling), target_batch_unrolling
# }}}

# Dataset class (oneshot_spectre){{{
    """
    input: spectrum (one channel from original data)
    output: onset prediction(0~1, 1channel)

    data_dir -> ddc's chart_onset/mel80hop441
    chartファイル生成後に入ってるtrain.txtとかは消しておく
    曲の長さがだいたい同じだけどまちまちだから、
        * バッチサイズ１(一曲ずつ読み込む)
        * サンプルサイズ＝一曲まるまる
    """

class Dataset_oneshot(data.Dataset):
    def __init__(self, data_dir, trim=False): #no trim for match timestep
        super(Dataset_oneshot, self).__init__()

        self.trim = trim

        self.root_path = data_dir
        self.filenames = [x for x in sorted(os.listdir(self.root_path)) if x.split('.')[-1] == 'pkl']

    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filenames[index])

        with open(filepath, 'rb') as f:
            file = pickle.load(f)

        return file

    def __len__(self):
        return len(self.filenames)


class DataLoader_oneshot(data.DataLoader):
    def __init__(self, data_dir, receptive_fields, ddc_channel_select,
                 batch_size=1, shuffle=True, valid=False):
        """
        DataLoader for WaveNet
        :param data_dir:
        :param receptive_fields: integer. size(length) of receptive fields
                            sample size has to be bigger than receptive fields.
                            |-- receptive field --|---------------------|
                            |------- samples -------------------|
                            |---------------------|-- outputs --|
        :param ddc_channel_select: select channel of ddc input melspectrogram.
        :param batch_size:
        :param shuffle:
        """
        # add validation feature. if valid=True, this is dataloader for validation
        datasets = []
        for d_dir in data_dir:
            dataset = Dataset_oneshot(d_dir)
            dataset_size = len(dataset)
            train_size = int(dataset_size*0.9)
            if valid==False:
                dataset = data.Subset(dataset, list(range(train_size)))
            else:
                dataset = data.Subset(dataset, list(range(train_size, dataset_size)))
            datasets.append(dataset)
        dataset = data.ConcatDataset(datasets)
        super(DataLoader_oneshot, self).__init__(dataset, batch_size, shuffle)

        # if sample_size <= receptive_fields:
        #     raise Exception("sample_size has to be bigger than receptive_fields")

        # self.sample_size = sample_size
        self.receptive_fields = receptive_fields
        self.ddc_channel_select = ddc_channel_select
        self.diffs = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']

        self.collate_fn = self._collate_fn

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    @staticmethod
    def _symbol_encode(sequence):
        # res.shape: [len(chart.onsets), 1]
        res = np.array([[int(symbol, base=4)] for symbol in sequence])
        return res

    def _collate_fn(self, files):
        file_chart = files[0]
        _, song_feat, charts = file_chart

        # song_feat
        song_feat = song_feat[:, :, self.ddc_channel_select] # slice selected channel
        song_feat = song_feat.reshape(-1, song_feat.shape[1]*song_feat.shape[2]) # [time, freq*channel]
        song_feat_batch = np.array([song_feat]) # batch_size:1
        song_feat_batch = np.pad(song_feat_batch, [[0, 0], [self.receptive_fields, 0], [0, 0]], 'constant')

        # chart
        target_batch_iter = []
        diff_batch_iter = []
        for chart in random.sample(charts, len(charts)):
            target_onsets = [[int(frame_idx in chart.onsets)] for frame_idx in range(chart.nframes)]
            target_onsets = np.array(target_onsets)
            target_symbol = self._symbol_encode(chart.sequence)
            target_batch = np.zeros((target_onsets.shape[0],1))
            target_batch[sorted(list(chart.onsets))] = target_symbol
            target_batch = np.concatenate([target_onsets, target_batch], axis=1)[np.newaxis]
            target_batch_iter.append(target_batch)
            diff = chart.get_coarse_difficulty()
            diff_batch = np.zeros((1,5))
            diff_batch[0][self.diffs.index(diff)] = 1.0
            diff_batch_iter.append(diff_batch)

        # target_batch_iterというリストの中に一曲のchartをすべて入れてイテレーションをするので、yieldを利用する
        # unrollingと名付けたリスト(len=1)でバッチを投げているのは、切り抜いてサンプルサイズ毎に学習する場合に合わせているから
        for target_batch, diff_batch in zip(target_batch_iter, diff_batch_iter):
            song_feat_batch_unrolling = [self._variable(song_feat_batch)]
            target_batch_unrolling = [self._variable(target_batch)]
            diff_batch_unrolling = [self._variable(diff_batch)]
            yield (song_feat_batch_unrolling, diff_batch_unrolling), target_batch_unrolling
# }}}
# Dataset class (oneshot_raw){{{
    """
    input: raw_audio
    output: onset prediction(0~1, 1channel)

    data_dir -> ddc's chart_onset/mel80hop441
    chartファイル生成後に入ってるtrain.txtとかは消しておく
    曲の長さがだいたい同じだけどまちまちだから、
        * バッチサイズ１(一曲ずつ読み込む)
        * サンプルサイズ＝一曲まるまる
    """

class Dataset_oneshot_raw(data.Dataset):
    def __init__(self, data_dir, sample_rate=16000, in_channels=256, trim=False): #no trim for match timestep
        super(Dataset_oneshot_raw, self).__init__()

        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.trim = trim

        self.root_path = data_dir
        self.filenames_chart = [x for x in sorted(os.listdir(self.root_path)) if x.split('.')[-1] == 'pkl']
        self.filenames_audio = [x for x in sorted(os.listdir(self.root_path)) if x.split('.')[-1] == 'ogg']

    def __getitem__(self, index):
        filepath_chart = os.path.join(self.root_path, self.filenames_chart[index])
        filepath_audio = os.path.join(self.root_path, self.filenames_audio[index])

        with open(filepath_chart, 'rb') as f:
            file_chart = pickle.load(f)

        raw_audio = load_audio(filepath_audio, self.sample_rate, self.trim)
        encoded_audio = mu_law_encode(raw_audio, self.in_channels)
        encoded_audio = one_hot_encode(encoded_audio, self.in_channels)

        item = (file_chart, encoded_audio)
        return item

    def __len__(self):
        return len(self.filenames_chart)


class DataLoader_oneshot_raw(data.DataLoader):
    def __init__(self, data_dir, receptive_fields,
                 sample_size=0, sample_rate=16000, chart_sample_rate=100, in_channels=256,
                 batch_size=1, shuffle=True, valid=False):
        """
        DataLoader for WaveNet
        :param data_dir:
        :param receptive_fields: integer. size(length) of receptive fields
                            sample size has to be bigger than receptive fields.
                            |-- receptive field --|---------------------|
                            |------- samples -------------------|
                            |---------------------|-- outputs --|
        :param ddc_channel_select: select channel of ddc input melspectrogram.
        :param batch_size:
        :param shuffle:
        """
        # add validation feature. if valid=True, this is dataloader for validation
        datasets = []
        for d_dir in data_dir:
            dataset = Dataset_oneshot_raw(d_dir)
            dataset_size = len(dataset)
            train_size = int(dataset_size*0.9)
            if valid==False:
                dataset = data.Subset(dataset, list(range(train_size)))
            else:
                dataset = data.Subset(dataset, list(range(train_size, dataset_size)))
            datasets.append(dataset)
        dataset = data.ConcatDataset(datasets)
        super(DataLoader_oneshot_raw, self).__init__(dataset, batch_size, shuffle)

        # if sample_size <= receptive_fields:
        #     raise Exception("sample_size has to be bigger than receptive_fields")

        if sample_rate % chart_sample_rate:
            raise Exception("sample_rate は 100の倍数にしてほしい")

        tmp = sample_size % (sample_rate / chart_sample_rate)
        if tmp != 0:
            self.sample_size = int(sample_size + (sample_rate / chart_sample_rate) - tmp)
            print("{}({}):sample_sizeがちょっとキリ悪いので増やしました。{}->{}".format(self.__class__.__name__,
                                                                                        ('valid' if valid else 'train'),
                                                                                        sample_size,
                                                                                        self.sample_size))
        else:
            self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.chart_sample_rate = chart_sample_rate

        self.receptive_fields = receptive_fields
        self.diffs = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']

        self.collate_fn = self._collate_fn

    def calc_sample_size(self, audio):
        if len(audio[0]) >= self.sample_size + self.receptive_fields:
            audio_sample_size = self.sample_size
            chart_sample_size = audio_sample_size / (self.sample_rate / self.chart_sample_rate)
        else:
            audio_sample_size = len(audio[0])
            chart_sample_size = (audio_sample_size - self.receptive_fields) / (self.sample_rate / self.chart_sample_rate)
            chart_sample_size = 0 if chart_sample_size < 0 else chart_sample_size
        assert chart_sample_size == int(chart_sample_size)
        return audio_sample_size, int(chart_sample_size)

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    @staticmethod
    def _symbol_encode(sequence):
        # res.shape: [len(chart.onsets), 1]
        res = np.array([[int(symbol, base=4)] for symbol in sequence])
        return res

    def _collate_fn(self, files):
        file = files[0]
        file_chart, encoded_audio = file
        _, _, charts = file_chart

        # song_feat
        padding = charts[0].nframes*(self.sample_rate/self.chart_sample_rate) - encoded_audio.shape[0]
        assert padding >= 0
        song_feat_batch = np.array([encoded_audio]) # batch_size=1
        song_feat_batch = np.pad(song_feat_batch, [[0,0],[self.receptive_fields,int(padding)],[0,0]], 'constant')

        # chart
        target_batch_iter = []
        diff_batch_iter = []
        for chart in random.sample(charts, len(charts)):
            target_onsets = [[int(frame_idx in chart.onsets)] for frame_idx in range(chart.nframes)]
            target_onsets = np.array(target_onsets)
            target_symbol = self._symbol_encode(chart.sequence)
            target_batch = np.zeros((target_onsets.shape[0],1))
            target_batch[sorted(list(chart.onsets))] = target_symbol
            target_batch = np.concatenate([target_onsets, target_batch], axis=1)[np.newaxis]
            target_batch_iter.append(target_batch)
            diff = chart.get_coarse_difficulty()
            diff_batch = np.zeros((1,5))
            diff_batch[0][self.diffs.index(diff)] = 1.0
            diff_batch_iter.append(diff_batch)

        # target_batch_iterというリストの中に一曲のchartをすべて入れてイテレーションをするので、yieldを利用する
        # raw_audioモデルではsample_sizeを制限すると曲を刻んでunrollingにする
        for target_batch, diff_batch in zip(target_batch_iter, diff_batch_iter):
            if self.sample_size:
                # サンプルサイズが設定されている場合はその長さに音源を切って順にトレーニングに投げる
                song_feat_batch_copy = song_feat_batch.copy()
                song_feat_batch_unrolling = []
                target_batch_unrolling = []
                diff_batch_unrolling = []
                audio_sample_size, chart_sample_size = self.calc_sample_size(song_feat_batch_copy)

                while audio_sample_size > self.receptive_fields:
                    inputs = song_feat_batch_copy[:, :audio_sample_size+self.receptive_fields, :]
                    targets = target_batch[:, :chart_sample_size]
                    song_feat_batch_unrolling.append(self._variable(inputs))
                    target_batch_unrolling.append(self._variable(targets))
                    diff_batch_unrolling.append(self._variable(diff_batch))

                    song_feat_batch_copy = song_feat_batch_copy[:, audio_sample_size:, :]
                    target_batch = target_batch[:, chart_sample_size:]
                    audio_sample_size, chart_sample_size = self.calc_sample_size(song_feat_batch_copy)

                yield (song_feat_batch_unrolling, diff_batch_unrolling), target_batch_unrolling

            else:
                # サンプルサイズが設定されていない場合はそのまま投げる(メモリ溢れても知らないよ)
                song_feat_batch_unrolling = [self._variable(song_feat_batch)]
                target_batch_unrolling = [self._variable(target_batch)]
                diff_batch_unrolling = [self._variable(diff_batch)]
                yield (song_feat_batch_unrolling, diff_batch_unrolling), target_batch_unrolling
# }}}
# Dataset class (oneshot_raw_snip){{{
    """
    input: raw_audio
    output: onset prediction(0~1, 1channel)

    data_dir -> ddc's chart_onset/mel80hop441
    chartファイル生成後に入ってるtrain.txtとかは消しておく
    曲の長さがだいたい同じだけどまちまちだから、
        * バッチサイズ１(一曲ずつ読み込む)
        * サンプルサイズ＝一曲まるまる
    """

class DataLoader_oneshot_raw_snap(data.DataLoader):
    def __init__(self, data_dir, receptive_fields,
                 sample_size=0, sample_rate=16000, chart_sample_rate=100, in_channels=256,
                 batch_size=1, shuffle=True, valid=False):
        """
        DataLoader for WaveNet
        :param data_dir:
        :param receptive_fields: integer. size(length) of receptive fields
                            sample size has to be bigger than receptive fields.
                            |-- receptive field --|---------------------|
                            |------- samples -------------------|
                            |---------------------|-- outputs --|
        :param ddc_channel_select: select channel of ddc input melspectrogram.
        :param batch_size:
        :param shuffle:
        """
        # add validation feature. if valid=True, this is dataloader for validation
        datasets = []
        for d_dir in data_dir:
            dataset = Dataset_oneshot_raw(d_dir)
            dataset_size = len(dataset)
            train_size = int(dataset_size*0.9)
            if valid==False:
                dataset = data.Subset(dataset, list(range(train_size)))
            else:
                dataset = data.Subset(dataset, list(range(train_size, dataset_size)))
            datasets.append(dataset)
        dataset = data.ConcatDataset(datasets)
        super(DataLoader_oneshot_raw_snap, self).__init__(dataset, batch_size, shuffle)

        # if sample_size <= receptive_fields:
        #     raise Exception("sample_size has to be bigger than receptive_fields")

        tmp = sample_size % (sample_rate / chart_sample_rate)
        if tmp != 0:
            self.sample_size = int(sample_size + (sample_rate / chart_sample_rate) - tmp)
            print("{}({}):sample_sizeがちょっとキリ悪いので増やしました。{}->{}".format(self.__class__.__name__,
                                                                                        ('valid' if valid else 'train'),
                                                                                        sample_size,
                                                                                        self.sample_size))
        else:
            self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.chart_sample_rate = chart_sample_rate

        self.receptive_fields = receptive_fields
        self.diffs = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']

        self.collate_fn = self._collate_fn
        self.valid = valid

    def calc_sample_size(self, audio):
        if len(audio[0]) >= self.sample_size + self.receptive_fields:
            audio_sample_size = self.sample_size
            chart_sample_size = audio_sample_size / (self.sample_rate / self.chart_sample_rate)
        else:
            audio_sample_size = len(audio[0])
            chart_sample_size = (audio_sample_size - self.receptive_fields) / (self.sample_rate / self.chart_sample_rate)
            chart_sample_size = 0 if chart_sample_size < 0 else chart_sample_size
        assert chart_sample_size == int(chart_sample_size)
        return audio_sample_size, int(chart_sample_size)

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    @staticmethod
    def _symbol_encode(sequence):
        # res.shape: [len(chart.onsets), 1]
        res = np.array([[int(symbol, base=4)] for symbol in sequence])
        return res

    def _collate_fn(self, files):
        file = files[0]
        file_chart, encoded_audio = file
        _, _, charts = file_chart

        # song_feat
        padding = charts[0].nframes*(self.sample_rate/self.chart_sample_rate) - encoded_audio.shape[0]
        assert padding >= 0
        song_feat_batch = np.array([encoded_audio]) # batch_size=1
        song_feat_batch = np.pad(song_feat_batch, [[0,0],[self.receptive_fields,int(padding)],[0,0]], 'constant')

        # chart
        target_batch_iter = []
        diff_batch_iter = []
        for chart in random.sample(charts, len(charts)):
            target_onsets = [[int(frame_idx in chart.onsets)] for frame_idx in range(chart.nframes)]
            target_onsets = np.array(target_onsets)
            target_symbol = self._symbol_encode(chart.sequence)
            target_batch = np.zeros((target_onsets.shape[0],1))
            target_batch[sorted(list(chart.onsets))] = target_symbol
            target_batch = np.concatenate([target_onsets, target_batch], axis=1)[np.newaxis]
            target_batch_iter.append(target_batch)
            diff = chart.get_coarse_difficulty()
            diff_batch = np.zeros((1,5))
            diff_batch[0][self.diffs.index(diff)] = 1.0
            diff_batch_iter.append(diff_batch)

        # target_batch_iterというリストの中に一曲のchartをすべて入れてイテレーションをするので、yieldを利用する
        # raw_audioモデルではsample_sizeを制限すると曲を刻んでunrollingにする
        # このsnipバージョンは刻み方を等間隔じゃなくてランダムにしてみた。こっちのほうが良いと思う
        for target_batch, diff_batch in zip(target_batch_iter, diff_batch_iter):
            if self.sample_size:
                # サンプルサイズが設定されている場合はその長さに切ったものを投げる。
                song_feat_batch_copy = song_feat_batch.copy()
                song_feat_batch_unrolling = []
                target_batch_unrolling = []
                diff_batch_unrolling = []
                audio_sample_size = self.sample_size
                input_scale = self.sample_rate // self.chart_sample_rate
                chart_sample_size = self.sample_size // input_scale

                # trainの際にランダムサンプル 切り出す回数を、適当に曲の長さ/サンプルサイズにする
                if not self.valid:
                    num_unrolling = int(song_feat_batch.shape[1]/audio_sample_size)
                    for nu in range(num_unrolling):
                        former = []
                        while True:
                            target_index_start = random.randint(0, target_batch.shape[1]-chart_sample_size)
                            if target_index_start not in former:
                                former.append(target_index_start)
                                break
                        targets = target_batch[:, target_index_start:target_index_start+chart_sample_size, :]
                        audio_index_start = target_index_start*input_scale
                        inputs = song_feat_batch_copy[:, audio_index_start:audio_index_start+audio_sample_size+self.receptive_fields, :]
                        song_feat_batch_unrolling.append(self._variable(inputs))
                        target_batch_unrolling.append(self._variable(targets))
                        diff_batch_unrolling.append(self._variable(diff_batch))
                    yield (song_feat_batch_unrolling, diff_batch_unrolling), target_batch_unrolling

                else:
                    while audio_sample_size > self.receptive_fields:
                        inputs = song_feat_batch_copy[:, :audio_sample_size+self.receptive_fields, :]
                        targets = target_batch[:, :chart_sample_size]
                        song_feat_batch_unrolling.append(self._variable(inputs))
                        target_batch_unrolling.append(self._variable(targets))
                        diff_batch_unrolling.append(self._variable(diff_batch))

                        song_feat_batch_copy = song_feat_batch_copy[:, audio_sample_size:, :]
                        target_batch = target_batch[:, chart_sample_size:]
                        audio_sample_size, chart_sample_size = self.calc_sample_size(song_feat_batch_copy)

                    yield (song_feat_batch_unrolling, diff_batch_unrolling), target_batch_unrolling

            else:
                # サンプルサイズが設定されていない場合はそのまま投げる(メモリ溢れても知らないよ)
                song_feat_batch_unrolling = [self._variable(song_feat_batch)]
                target_batch_unrolling = [self._variable(target_batch)]
                diff_batch_unrolling = [self._variable(diff_batch)]
                yield (song_feat_batch_unrolling, diff_batch_unrolling), target_batch_unrolling
# }}}
# Dataset class (oneshot_spectre_snap){{{
    """
    input: spectrum (one channel from original data)
    output: onset prediction(0~1, 1channel)

    data_dir -> ddc's chart_onset/mel80hop441
    chartファイル生成後に入ってるtrain.txtとかは消しておく
    曲の長さがだいたい同じだけどまちまちだから、
        * バッチサイズ１(一曲ずつ読み込む)
    """

class DataLoader_oneshot_snap(data.DataLoader):
    def __init__(self, data_dir, receptive_fields, ddc_channel_select,
                 sample_size=0,
                 batch_size=1, shuffle=True, valid=False):
        """
        DataLoader for WaveNet
        :param data_dir:
        :param receptive_fields: integer. size(length) of receptive fields
                            sample size has to be bigger than receptive fields.
                            |-- receptive field --|---------------------|
                            |------- samples -------------------|
                            |---------------------|-- outputs --|
        :param ddc_channel_select: select channel of ddc input melspectrogram.
        :param batch_size:
        :param shuffle:
        """
        # dataset
        datasets = []
        for d_dir in data_dir:
            dataset = Dataset_oneshot(d_dir)
            dataset_size = len(dataset)
            train_size = int(dataset_size*0.9)
            if valid==False:
                dataset = data.Subset(dataset, list(range(train_size)))
            else:
                dataset = data.Subset(dataset, list(range(train_size, dataset_size)))
            datasets.append(dataset)
        dataset = data.ConcatDataset(datasets)
        super(DataLoader_oneshot_snap, self).__init__(dataset, batch_size, shuffle)

        # if sample_size <= receptive_fields:
        #     raise Exception("sample_size has to be bigger than receptive_fields")

        self.sample_size = sample_size
        self.receptive_fields = receptive_fields
        self.ddc_channel_select = ddc_channel_select
        self.diffs = ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']
        for d_dir in data_dir:
            # self.author_dict = load_id_dict(pathlib.Path(d_dir)/'labels'/'freetext_to_id.csv')
            self.author_dict = load_id_dict(os.path.join(d_dir,'labels','freetext_to_id.csv'))

        self.collate_fn = self._collate_fn
        self.valid = valid

    def calc_sample_size(self, audio):
        if len(audio[0]) >= self.sample_size + self.receptive_fields:
            tmp_sample_size = self.sample_size
        else:
            tmp_sample_size = len(audio[0]) # この時len(audio[0])がゼロならば終了
        return tmp_sample_size

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)

    @staticmethod
    def _symbol_encode(sequence):
        # res.shape: [len(chart.onsets), 1]
        res = np.array([[int(symbol, base=4)] for symbol in sequence])
        return res

    def _collate_fn(self, files):
        file_chart = files[0]
        _, song_feat, charts = file_chart

        # song_feat
        song_feat = song_feat[:, :, self.ddc_channel_select] # slice selected channel
        song_feat = song_feat.reshape(-1, song_feat.shape[1]*song_feat.shape[2]) # [time, freq*channel]
        song_feat_batch = np.array([song_feat], dtype=np.float32) # batch_size:1
        song_feat_batch = np.pad(song_feat_batch, [[0, 0], [self.receptive_fields, 0], [0, 0]], 'constant')

        # chart
        target_batch_iter = []
        gc_batch_iter = []
        for chart in random.sample(charts, len(charts)):
            target_onsets = [[int(frame_idx in chart.onsets)] for frame_idx in range(chart.nframes)]
            target_onsets = np.array(target_onsets, dtype=np.float32)
            target_symbol = self._symbol_encode(chart.sequence)
            target_batch = np.zeros((target_onsets.shape[0],1), dtype=np.float32)
            target_batch[sorted(list(chart.onsets))] = target_symbol
            target_batch = np.concatenate([target_onsets, target_batch], axis=1)[np.newaxis]
            target_batch_iter.append(target_batch)
            diff = chart.get_coarse_difficulty()
            diff_batch = np.zeros((1,5), dtype=np.float32)
            diff_batch[0][self.diffs.index(diff)] = 1.0
            author = chart.get_freetext()
            author_batch = np.zeros((1,12), dtype=np.float32)
            author_batch[0][self.author_dict[author]] = 1.0
            gc_batch = np.hstack((diff_batch, author_batch))
            gc_batch_iter.append(gc_batch)

        # target_batch_iterというリストの中に一曲のchartをすべて入れてイテレーションをするので、yieldを利用する
        for target_batch, gc_batch in zip(target_batch_iter, gc_batch_iter):
            if self.sample_size != 0:
                song_feat_batch_copy = song_feat_batch.copy()
                song_feat_batch_unrolling = []
                target_batch_unrolling = []
                gc_batch_unrolling = []
                # make train unrollings
                if not self.valid:
                    # random sample by constant sample size
                    num_unrolling = int(song_feat_batch.shape[1]/self.sample_size)
                    assert num_unrolling > 0
                    for nu in range(num_unrolling):
                        former = []
                        while True:
                            target_index_start = random.randint(0, target_batch.shape[1]-self.sample_size)
                            if target_index_start not in former:
                                former.append(target_index_start)
                                break
                        targets = target_batch[:, target_index_start:target_index_start+self.sample_size, :]
                        audio_index_start = target_index_start
                        inputs = song_feat_batch_copy[:, audio_index_start:audio_index_start+self.sample_size+self.receptive_fields, :]
                        song_feat_batch_unrolling.append(self._variable(inputs))
                        target_batch_unrolling.append(self._variable(targets))
                        gc_batch_unrolling.append(self._variable(gc_batch))
                    yield (song_feat_batch_unrolling, gc_batch_unrolling), target_batch_unrolling
                # make valid unrollings
                else:
                    tmp_sample_size = self.sample_size
                    while tmp_sample_size > self.receptive_fields:
                        inputs = song_feat_batch_copy[:, :tmp_sample_size+self.receptive_fields, :]
                        targets = target_batch[:, :tmp_sample_size]
                        song_feat_batch_unrolling.append(self._variable(inputs))
                        target_batch_unrolling.append(self._variable(targets))
                        gc_batch_unrolling.append(self._variable(gc_batch))

                        song_feat_batch_copy = song_feat_batch_copy[:, tmp_sample_size:, :]
                        target_batch = target_batch[:, tmp_sample_size:]
                        tmp_sample_size = self.calc_sample_size(song_feat_batch_copy)

                    yield (song_feat_batch_unrolling, gc_batch_unrolling), target_batch_unrolling

            # input whole data
            else:
                song_feat_batch_unrolling = [self._variable(song_feat_batch)]
                target_batch_unrolling = [self._variable(target_batch)]
                gc_batch_unrolling = [self._variable(gc_batch)]
                yield (song_feat_batch_unrolling, gc_batch_unrolling), target_batch_unrolling
# }}}
