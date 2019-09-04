# Introduction
This algorithm is one of Dynamic-Time-Warping(DTW) which can adjust Multiple sequences.

Currently, DTW uses only 2 sequences.
For example, let reference sequence is A (shape=[300, 100]), and comparison sequences are B (shape=[250, 100]), C (shape=[500, 100]).
If we use DTW to [A, B] and [A,C], each DTW result is different length. ([A, B] = 350, [A, C] = 520). So it's hard to input the model.
The Reason why I make this is to adjust the length of the references spectrogram and the various comparison spectrograms as the reference spectrogram.

Basic theory is euclidean distance and DTW.

# Installation
```
git clone https://github.com/kaen2891/Multi_DTW.git
cd Multi_DTW
python3 test_Multi_DTW.py
```

# How to use
In test_Multi_DTW.py, there 2 ways to use Multi-DTW

One is 1-d array. You must make it to 2-d.
```
a = [1, 6, 2, 3, 0, 9, 4, 3, 6, 3]
c = [1, 3, 4, 9, 8, 2, 1]
a = np.expand_dims(a, axis=1)
c = np.expand_dims(c, axis=1)

dtw_a_c = Multi_DTW.dtw(a, c, len(a), len(c))

print(dtw_a_c)
```

The other is 2-d array. In code, I use spectrogram
```
ref_man_speech = './test_wav/man_reference.wav'
man_speech2 = './test_wav/man_comparison.wav'

r_m, sr = librosa.load(ref_man_speech, sr=16000)
c_m, sr = librosa.load(man_speech2, sr=16000)

r_m_spec = librosa.stft(r_m, n_fft=nfft, hop_length=hop)
c_m_spec = librosa.stft(c_m, n_fft=nfft, hop_length=hop)

r_m_c_m_dtw = Multi_DTW.dtw(r_m_spec.T, c_m_spec.T, len(r_m_spec.T), len(c_m_spec.T))
cm_dtw = c_m_spec[:, r_m_c_m_dtw]

print(np.shape(cm_dtw))
```

# Test dataset
Test dataset is in folder (test_wav)

# Requirements
* [**librosa**](https://librosa.github.io/librosa/) (if you want to use spectrogram for DTW)
* **Python** version >= 3.5
* **matplotlib** (if you want to plot result)
* **numpy**


# Author
June-Woo Kim, ABR Lab in Kyungpook National University **kaen2891@gmail.com**

More infor about speech, please visit my [**blog**](https://kaen2891.tistory.com/)

