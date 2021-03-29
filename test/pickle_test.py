import numpy, pickle, os, sys
import librosa
import torch, torchaudio
import time
from IPython import embed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wavenet.utils.chart import OnsetChart, SymbolicChart

# fp = '../datasets/ddc_onset/fraxtil/Fraxtil_sArrowArrangements_Girls.pkl'
# with open(fp, 'rb') as f:
# 	file = pickle.load(f)

fp_audio = '../datasets/raw_audio_tracks/original_ogg/fraxtil/Fraxtil_sArrowArrangements_Girls.ogg'
fp_audio_wav = '../datasets/raw_audio_tracks/wav_mono_16000hz/fraxtil/Fraxtil_sArrowArrangements_Girls.wav'
resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)

start = time.time()
audio_torch, sr = torchaudio.load(fp_audio)
audio_torch = resample(audio_torch)
audio_torch = torch.mean(audio_torch, 0)
end_torch = time.time() - start
start = time.time()
audio_torch_wav, sr = torchaudio.load_wav(fp_audio_wav)
end_torch = time.time() - start
start = time.time()
audio, _ = librosa.load(fp_audio, sr=16000, mono=True)
end = time.time() - start
# audio = audio.reshape(-1, 1)

# for chart in file[2]:
#     print(chart.get_coarse_difficulty())
embed()
exit()
# mean_song_feature = numpy.mean(song_feature, axis=0)
# print(mean_song_feature)
# test = [item for sublist in [
# 							 [[[1,2,3],[4,5,6],[7,8,9]],
# 						      [[1,2,3],[4,5,6],[7,8,9]]]
# 						     						    ,
# 							 [[[1,2,3],[4,5,6],[7,8,9]],
# 						      [[1,2,3],[4,5,6],[7,8,9]]]
# 						     						    ]for item in sublist]
# test = [item for sublist in [[1,2,3,4,5,6,7,8,9],
#                              [1,2,3,4,5,6,7,8,9]]for item in sublist]
# print(test)
