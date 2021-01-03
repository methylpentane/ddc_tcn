"""
Test mu-law encoding and decoding
"""

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wavenet.utils.data import DataLoader, DataLoader_oneshot


RECEPTIVE_FIELDS = 1000
SAMPLE_SIZE = 2000
SAMPLE_RATE = 8000
IN_CHANNELS = 256
DDC_CHANNEL_SELECT = [0]
TEST_AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_AUDIO_DIR_ONESHOT = os.path.join(os.path.dirname(__file__), '../datasets/oneshot/fraxtil')


def test_data_loader():
    data_loader = DataLoader(TEST_AUDIO_DIR,
                             RECEPTIVE_FIELDS, SAMPLE_SIZE, SAMPLE_RATE, IN_CHANNELS,
                             shuffle=False)

    dataset_size = []

    for dataset in data_loader:
        input_size = []
        target_size = []

        for i, t in dataset:
            input_size.append(i.shape)
            target_size.append(t.shape)

        dataset_size.append([input_size, target_size])

    assert dataset_size[0][0][0] == torch.Size([1, 2000, 256])
    assert dataset_size[0][1][0] == torch.Size([1, 1000])
    assert dataset_size[0][0][-1] == torch.Size([1, 1839, 256])
    assert dataset_size[0][1][-1] == torch.Size([1, 839])

    assert dataset_size[1][0][0] == torch.Size([1, 2000, 256])
    assert dataset_size[1][1][0] == torch.Size([1, 1000])
    assert dataset_size[1][0][-1] == torch.Size([1, 1762, 256])
    assert dataset_size[1][1][-1] == torch.Size([1, 762])

    assert len(dataset_size[0][0]) == 8
    assert len(dataset_size[1][0]) == 8

def test_oneshot_dataloader():
    data_loader = DataLoader_oneshot(TEST_AUDIO_DIR_ONESHOT, RECEPTIVE_FIELDS, DDC_CHANNEL_SELECT)
    for dataset in data_loader:
        for inputs, targets in dataset:
            inputs, gc_inputs = inputs
            from IPython import embed
            embed()
            exit()

if __name__ == "__main__":
    test_oneshot_dataloader()
