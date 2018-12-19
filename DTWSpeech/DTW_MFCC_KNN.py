""" 
@author: zoutai
@file: DTW_MFCC_KNN.py 
@time: 2018/12/12 
@description: 
"""

import time

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from DTWSpeech.dtw import dtw

y1, sr1 = librosa.load('train/fcmc0-a1-t.wav')
y2, sr2 = librosa.load('train/fcmc0-b1-t.wav')

plt.subplot(1, 2, 1)
mfcc1 = librosa.feature.mfcc(y1, sr1)
librosa.display.specshow(mfcc1)

plt.subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y2, sr2)
librosa.display.specshow(mfcc2)

# Calculate the DTW between the 2 sample audios 'a' and 'b'
dist, cost, path, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
print('Normalized distance between the two sounds:', dist)

import os

dirname = "train"
files = [f for f in os.listdir(dirname) if not f.startswith('.')]

start = time.clock()
minval = 200
distances = np.ones((len(files), len(files)))
y = np.ones(len(files))

for i in range(len(files)):
    y1, sr1 = librosa.load(dirname + "/" + files[i])
    mfcc1 = librosa.feature.mfcc(y1, sr1)
    for j in range(len(files)):
        y2, sr2 = librosa.load(dirname + "/" + files[j])
        mfcc2 = librosa.feature.mfcc(y2, sr2)
        dist, _, _, _ = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
        #         print files[i],mfcc1.T[0][0],mfcc2.T[0][0],files[j],dist
        #         if dist<minval:
        #             minval = dist
        distances[i, j] = dist
    if i % 2 == 0:
        y[i] = 0  # 'a'
    else:
        y[i] = 1  # 'b'
print("Time used: {}s".format(time.clock() - start))

print(distances[0])

# a = 155.156
# b = 184.702
# c = 158.231
label = ['a', 'b']

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
classifier.fit(distances, y)

y, sr = librosa.load('test/farw0-b1-t.wav')
mfcc = librosa.feature.mfcc(y, sr)
distanceTest = []
for i in range(len(files)):
    y1, sr1 = librosa.load(dirname + "/" + files[i])
    mfcc1 = librosa.feature.mfcc(y1, sr1)
    dist, _, _, _ = dtw(mfcc.T, mfcc1.T, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
    distanceTest.append(dist)

# pre = classifier.predict(distanceTest)[0] # False
pre = classifier.predict([distanceTest])[0]
print(pre)

label[int(pre)]

print("Predict audio is: '{}'".format(label[int(pre)]))
plt.show()
