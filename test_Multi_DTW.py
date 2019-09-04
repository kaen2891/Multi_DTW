import os
import numpy as np
import Multi_DTW

# test 1d array
a = [1, 6, 2, 3, 0, 9, 4, 3, 6, 3]
b = [1, 3, 4, 9, 8, 2, 1, 5, 7, 3]
c = [1, 3, 4, 9, 8, 2, 1]
d = [8, 0, 9, 4, 8, 5, 8, 10]
e = [3, 2, 5, 4, 7, 6, 2, 1, 8, 9, 3, 2, 1]

a = np.expand_dims(a, axis=1)
b = np.expand_dims(b, axis=1)
c = np.expand_dims(c, axis=1)
d = np.expand_dims(d, axis=1)
e = np.expand_dims(e, axis=1)

dtw_a_b = Multi_DTW.dtw(a, b, len(a), len(b))
dtw_a_c = Multi_DTW.dtw(a, c, len(a), len(c))
dtw_a_d = Multi_DTW.dtw(a, d, len(a), len(d))
dtw_a_e = Multi_DTW.dtw(a, e, len(a), len(e))

#print(a)
print(dtw_a_b)
print(dtw_a_c)
print(dtw_a_d)
print(dtw_a_e)

print("all dtw shape is, a_b:{}, a_c:{}, a_d:{}, a_e:{}".format(np.shape(dtw_a_b), np.shape(dtw_a_c), np.shape(dtw_a_d), np.shape(dtw_a_e)))
print("all dtw result, ab_dtw:{}, ac_dtw:{}, ad_dtw:{}, ae_dtw:{}".format(dtw_a_b, dtw_a_c, dtw_a_d, dtw_a_e))


# test real speech (2d spectrogram) 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import librosa.display

def return_mag_pha(input_stft):
    mag, pha = librosa.magphase(input_stft)
    return mag, pha

def plot_wav(spectrogram, name, save_dir):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram,ref=np.max), y_axis='hz', x_axis='time', sr=16000, hop_length=hop)
        
    plt.title(name)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_dir + name + '.png')

nfft = 512 # number of fft, each frame length is 32ms
hop = 256 # number of noverlap, each noverlap length is 16ms

ref_man_speech = './test_wav/man_reference.wav'
man_speech2 = './test_wav/man_comparison.wav'
woman_speech1 = './test_wav/woman_comparison1.wav'
woman_speech2 = './test_wav/woman_comparison2.wav'

r_m, sr = librosa.load(ref_man_speech, sr=16000)
c_m, sr = librosa.load(man_speech2, sr=16000)
c_w1, sr = librosa.load(woman_speech1, sr=16000)
c_w2, sr = librosa.load(woman_speech2, sr=16000)

r_m_spec = librosa.stft(r_m, n_fft=nfft, hop_length=hop)
#print("r_m_spec shape",np.shape(r_m_spec))

c_m_spec = librosa.stft(c_m, n_fft=nfft, hop_length=hop)
#print("c_m_spec",np.shape(c_m_spec))
r_m_c_m_dtw = Multi_DTW.dtw(r_m_spec.T, c_m_spec.T, len(r_m_spec.T), len(c_m_spec.T))
#print("r_m_c_m spec",np.shape(r_m_c_m_dtw))
cm_dtw = c_m_spec[:, r_m_c_m_dtw]
#print("cm_dtw",np.shape(cm_dtw))

c_w1_spec = librosa.stft(c_w1, n_fft=nfft, hop_length=hop)
#print("c_w1_spec", np.shape(c_w1_spec))
r_m_c_w1_dtw = Multi_DTW.dtw(r_m_spec.T, c_w1_spec.T, len(r_m_spec.T), len(c_w1_spec.T))
#print("r_m_c_w1_dtw",np.shape(r_m_c_w1_dtw))
cw1_dtw = c_w1_spec[:, r_m_c_w1_dtw]
#print("cw1_dtw",np.shape(cw1_dtw))

c_w2_spec = librosa.stft(c_w2, n_fft=nfft, hop_length=hop)
#print("c_w2_spec", np.shape(c_w2_spec))
r_m_c_w2_dtw = Multi_DTW.dtw(r_m_spec.T, c_w2_spec.T, len(r_m_spec.T), len(c_w2_spec.T))
#print("r_m_c_w2_dtw", np.shape(r_m_c_w2_dtw))
cw2_dtw = c_w2_spec[:, r_m_c_w2_dtw]
#print("cw2_dtw", np.shape(cw2_dtw))

print("reference spectrogram shape {} comparison man spectrogram shape {} comparison woman1 spectrogram shape {} comparison woman2 spectrogram shape {}".format(np.shape(r_m_spec), np.shape(c_m_spec), np.shape(c_w1_spec), np.shape(c_w2_spec)))
        
print("dtw comparison man spectrogram shape {} dtw comparison woman1 spectrogram shape {} dtw comparison woman2 spectrogram shape {}".format(np.shape(cm_dtw), np.shape(cw1_dtw), np.shape(cw2_dtw)))

### plot dtw result###
save_dir ='./test_wav/'
plot_wav(r_m_spec, 'reference', save_dir)
plot_wav(cm_dtw, 'dtw_comparison_man1', save_dir)
plot_wav(cw1_dtw, 'dtw_comparison_woman1', save_dir)
plot_wav(cw2_dtw, 'dtw_comparison_woman2', save_dir)
