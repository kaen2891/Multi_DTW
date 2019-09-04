# Introduction
This algorithm is one of Dynamic-Time-Warping(DTW) which can adjust Multiple sequences.

Currently, DTW uses only 2 sequences.
For example, let reference sequence is A (shape=[300, 100]), and comparison sequences are B (shape=[250, 100]), C (shape=[500, 100]).
If we use DTW to [A, B] and [A,C], each DTW result is different length. ([A, B] = 350, [A, C] = 520). So it's hard to input the model.
The Reason why I make this is to adjust the length of the references spectrogram and the various comparison spectrograms as the reference spectrogram.

Basic theory is euclidean distance and DTW.


# How to use
In code, there 2 ways to use Multi-DTW

One is 1-d array. You must make it to 2-d.

The other is 2-d array. In code, I use spectrogram

# Test dataset
Test dataset is in folder (test_wav)

# Requirements
* [**librosa**](https://librosa.github.io/librosa/) (if you want to use spectrogram for DTW)
* **Python** version >= 3.5
* [**matplotlib**] (if you want to plot result)
* [**numpy**]


# Author
June-Woo Kim, ABR Lab in Kyungpook National University **kaen2891@gmail.com**

more infor, please visit my [**blog**](https://kaen2891.tistory.com/)

