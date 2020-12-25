import numpy, pickle, os, sys
import librosa
from IPython import embed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from wavenet.utils.chart import OnsetChart, SymbolicChart

fp = '../datasets/ddc_onset/fraxtil/Fraxtil_sArrowArrangements_Girls.pkl'
fp_audio = '../datasets/ddc_onset/fraxtil/Fraxtil_sArrowArrangements_Girls.ogg'
with open(fp, 'rb') as f:
	file = pickle.load(f)
audio, _ = librosa.load(fp_audio, sr=16000, mono=True)
audio = audio.reshape(-1, 1)

#print("pickle file as \'file\'")
#for chart in file[2]:
#    print(chart.get_coarse_difficulty())
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
