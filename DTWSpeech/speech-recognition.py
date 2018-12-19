""" 
@author: zoutai
@file: speech-recognition.py 
@time: 2018/12/11 
@description: 
"""

import numpy as np

with open('./sounds/wavToTag.txt') as f:
    labels = np.array([l.replace('\n', '') for l in f.readlines()])

print(set(labels))

import librosa
import time

mfccs = {}
start = time.clock()

for i in range(len(labels)):
    y, sr = librosa.load('./sounds/{}.wav'.format(i))
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=13)
    mfccs[i] = mfcc.T
print("Time used: %fs." % (time.clock() - start))


def generate_train_test_set(P):
    train = []
    test = []

    for s in set(labels):
        all = np.find(labels == s)
        np.random.shuffle(all)
        train += all[:-P].tolist()
        test += all[-P:].tolist()

    return train, test


from DTWSpeech.dtw import dtw

# We use DP to speed up multiple tests
D = np.ones((len(labels), len(labels))) * -1


def cross_validation(train, test):
    score = 0.0

    for i in test:
        x = mfccs[i]

        dmin, jmin = float('inf'), -1
        for j in train:
            y = mfccs[j]

            d = D[i, j]
            if d == -1:
                d, _, _, _ = dtw(x, y, dist=lambda x, y: np.norm(x - y, ord=1))
                D[i, j] = d

            if d < dmin:
                dmin = d
                jmin = j

        score += 1.0 if (labels[i] == labels[jmin]) else 0.0

    return score / len(test)


start = time.clock()
train, test = generate_train_test_set(P=1)
rec_rate = cross_validation(train, test)
print('Recognition rate {}%'.format(100. * rec_rate))
print("Time used: %fs" % (time.clock() - start))

P = range(1, 10)
N = 5

rec = []

for p in P:
    r = [cross_validation(*generate_train_test_set(p)) for _ in range(N)]
    rec.append(r)

rec = np.array(rec)
rec = rec.reshape((N, -1))

import matplotlib.pyplot as plt

plt.errorbar(P - 0.5, np.mean(rec, axis=0), yerr=np.std(rec, axis=0))
plt.xticks(P - 0.5, P)
plt.ylim(0, 1)
print("Time used: %fs" % (time.clock() - start))
