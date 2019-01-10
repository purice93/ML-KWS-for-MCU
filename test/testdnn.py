""" 
@author: zoutai
@file: testdnn.py 
@time: 2018/12/17 
@description: 
"""

import tensorflow as tf


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    # 1读取文件频谱
    filename = '/home/zoutai/DataSets/speech_dataset/yes/ffd2ba2f_nohash_4.wav'
    # wav_filename_placeholder = filename
    # wav_loader = io_ops.read_file(wav_filename_placeholder)
    # wav_decoder = contrib_audio.decode_wav(wav_loader, desired_channels=1)
    #
    # # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    # spectrogram = contrib_audio.audio_spectrogram(
    #     wav_decoder,
    #     window_size=40,
    #     stride=40,
    #     magnitude_squared=True)
    # mfcc = contrib_audio.mfcc(
    #     spectrogram,
    #     wav_decoder.sample_rate,
    #     dct_coefficient_count=10)

    import librosa
    import python_speech_features as features

    import numpy as np
    y, sr = librosa.load(filename, sr=None)
    # Extract Spectrum of audio inputs
    # mfcc = features.mfcc(y, samplerate=sr, numcep=10, winlen=0.04, winstep=0.04)

    mfcc = [[-3.4194942e+01, 2.9670470e+00, 1.1738914e+00, 9.7612846e-01,
             6.9486332e-01, -5.9059840e-01, -6.3276869e-01, -9.2864978e-01,
             -1.2233410e+00, 1.8440172e-01, -3.4235214e+01, 1.9850335e+00,
             3.2335405e+00, 1.1455367e-01, 1.3445035e+00, 1.0579168e+00,
             4.1412225e-01, -1.0973158e+00, -2.1630844e-01, 3.3866054e-01,
             -6.6271887e+00, 2.7430875e+00, 9.9650402e+00, 4.6783864e-01,
             -8.9482874e-01, -1.1118565e+00, 8.5644609e-01, 4.4913971e-01,
             -8.5571128e-01, 1.1029941e+00, 8.4124870e+00, 2.1112235e+00,
             4.6162171e+00, 1.8334599e-02, -1.5343192e+00, -1.3558217e+00,
             2.8946385e-01, 5.8896166e-01, 3.3869785e-01, 8.8569957e-01,
             1.0218686e+01, 2.8520651e+00, 2.2554955e+00, -2.1705335e-01,
             -1.2264476e+00, -2.0060554e+00, 3.1681114e-01, 6.3004851e-01,
             -3.1428081e-01, -3.3595768e-01, 5.0316195e+00, 2.3689992e+00,
             2.5967171e+00, 2.0880902e-01, -5.0821453e-01, -2.4838262e+00,
             -2.2918399e-01, 1.4075452e-01, -1.8876675e-01, 5.5664092e-01,
             3.8237727e-01, -3.1912065e-01, 3.2968030e+00, -1.6188561e+00,
             2.1625315e-01, -1.9678675e+00, 5.9333688e-01, -9.5947152e-01,
             8.3070463e-03, 9.6953819e-03, -4.3058391e+00, -1.9131813e+00,
             2.5951560e+00, -2.2353518e+00, 4.9881661e-01, -1.7043195e+00,
             -9.3152356e-01, -9.0985626e-01, 7.8798592e-01, -6.4726162e-01,
             -5.7144041e+00, -3.0498528e+00, 6.0398042e-01, -3.0828445e+00,
             1.0721518e-01, -2.1091366e+00, -4.0878054e-01, -1.3729374e+00,
             -4.1109949e-01, -1.1415074e+00, -6.3667650e+00, -4.3073187e+00,
             -2.0500551e-01, -3.8324523e+00, -4.5246527e-01, -2.6610785e+00,
             9.3394242e-02, -8.9756584e-01, 7.0869482e-01, 1.2761207e-01,
             -8.1728849e+00, -4.3877850e+00, -5.4449427e-01, -3.3331103e+00,
             -1.8235508e-01, -2.1802266e+00, 4.3332863e-01, -3.1660363e-01,
             1.6604729e+00, -2.1165040e-01, -1.0727939e+01, -5.8247099e+00,
             -6.7493099e-01, -3.0871944e+00, 1.5487842e-01, -2.5797038e+00,
             7.5490522e-01, -2.7014714e-01, 6.4470047e-01, 1.0553528e+00,
             -1.0665574e+01, -4.6849442e+00, -1.0219340e+00, -2.9032230e+00,
             5.6179315e-01, -2.1752625e+00, 7.3237550e-01, 7.9258509e-02,
             9.1819793e-01, -7.6047319e-01, -1.1132480e+01, -2.9136889e+00,
             1.5116587e-01, -3.9473009e+00, 3.8955724e-01, -1.8017881e+00,
             2.4524249e-01, 6.6705608e-01, 6.0390586e-01, 2.0516028e-01,
             -1.8111385e+01, -1.7316824e+00, -4.7711152e-01, -3.3471732e+00,
             1.1049268e+00, -1.8103307e+00, 1.4052272e+00, 8.1470215e-01,
             3.1851465e-01, -3.0301306e-01, -2.5646849e+01, -4.7612956e-01,
             2.0604200e+00, -2.1651514e+00, 1.5088116e+00, -6.6252673e-01,
             9.2536390e-01, -4.0444070e-01, -4.9660242e-01, -1.0705365e+00,
             -3.0895529e+01, -1.7374872e-01, 1.9796300e+00, -6.4035445e-01,
             8.7914217e-01, -2.0637028e+00, -7.7086478e-02, 2.1001364e-01,
             -1.3147320e-01, -5.3195238e-01, -3.2669144e+01, 8.8499147e-01,
             1.8246269e+00, 1.9824359e-01, 9.2849040e-01, -2.1046505e+00,
             2.7661970e-01, -3.1248072e-01, 9.1331100e-01, 3.7230561e-03,
             -3.4183338e+01, 1.9311705e+00, 2.7224982e+00, 1.2209033e+00,
             2.7463260e-01, -1.0774504e+00, 3.0603120e-01, -7.9098499e-01,
             -3.0368215e-01, -8.3064646e-01, -3.4927601e+01, 2.7831223e+00,
             2.2212753e+00, 6.1375004e-01, -8.2468353e-02, -7.7749813e-01,
             -2.8016058e-01, -5.1618266e-01, 7.8253573e-01, 8.7826473e-01,
             -3.4972900e+01, 2.4085655e+00, 2.4627035e+00, 1.3345511e+00,
             7.2002620e-01, -7.5407207e-01, 2.6558584e-01, 7.1007752e-01,
             5.9068692e-01, 2.7767971e-01, -3.4761887e+01, 3.0384941e+00,
             1.9300233e+00, 4.0490535e-01, 2.9795719e-02, -1.3599268e+00,
             -1.2115438e+00, -1.2430253e-01, -7.0202702e-01, -3.9011011e-01,
             -3.5388111e+01, 1.8847787e+00, 4.3788052e-01, -1.5748229e-03,
             7.2207010e-01, 5.3366804e-01, -2.0432226e-01, -2.8813526e-01,
             -1.2917168e-01, 9.2518330e-02, -3.5297760e+01, 2.5215549e+00,
             2.4356098e+00, 9.1934365e-01, 5.9869635e-01, 4.9872756e-01,
             -2.2248250e-01, -3.9053787e-02, -4.1172618e-01, -4.3570751e-01,
             -3.5949562e+01, 2.5223236e+00, 2.3340530e+00, 1.0358025e+00,
             -8.4096573e-02, -3.9708334e-01, 4.6383935e-01, -2.9870054e-01,
             4.6251819e-01, -1.6407800e-01]]
    # mfcc = features.mfcc(y, samplerate=sr, numcep=10, nfilt=40, nfft=512, lowfreq=20,highfreq=4000,winlen=0.04, winstep=0.04)

    # tf.contrib.signal.stft(
    #     y,
    #     40,
    #     40,
    #     fft_length=None,
    #     pad_end=False,
    #     name=None
    # )
    # # 源码
    # wav_data_placeholder = filename
    # decoded_sample_data = contrib_audio.decode_wav(
    #     wav_data_placeholder,
    #     desired_channels=1,
    #     desired_samples=16000,
    #     name='decoded_sample_data')
    # spectrogram = contrib_audio.audio_spectrogram(
    #     decoded_sample_data.audio,
    #     window_size=40 * 16,
    #     stride=40 * 16,
    #     magnitude_squared=True)
    # mfcc = contrib_audio.mfcc(
    #     spectrogram,
    #     16000,
    #     dct_coefficient_count=10)
    #
    # # mfcc = tf.constant(mfcc, tf.float32)
    # # fingerprint_input = tf.convert_to_tensor(mfcc)
    # sess.run(tf.global_variables_initializer())
    # sess = tf.Session()
    # mfcc = mfcc.eval(session=sess)
    fingerprint_input = np.reshape(mfcc, [1, 250])
    # fingerprint_input = fingerprint_input.astype(np.float32)
    flow = fingerprint_input
    tf.summary.histogram('input', flow)
    fingerprint_size = 250
    layer_dim = [fingerprint_size]
    layer_dim.extend([144, 144, 144])
    flow = fingerprint_input
    tf.summary.histogram('input', flow)

    arr = []
    with open('/home/zoutai/code/ML-KWS-for-MCU/dnn_weights2.h', 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            line = line.split('{')[-1]
            line = line.split('}')[0]
            odom = line.strip().split(', ')  # 将单个数据分隔开存好
            numbers_float = np.array(list(map(float, odom)))  # 转化为浮点数
            arr.append(numbers_float)

    fc1_W_0, fc1_b_0, fc2_W_0, fc2_b_0, fc3_W_0, fc3_b_0, final_fc_0, Variable_0 = arr
    fc1_W_0 = np.reshape(fc1_W_0, [144, 250])
    fc1_W_0 = fc1_W_0.T
    # fc1_W_0 = np.reshape(fc1_W_0, [250, 144])
    fc2_W_0 = np.reshape(fc2_W_0, [144, 144])
    fc2_W_0 = fc2_W_0.T
    fc3_W_0 = np.reshape(fc3_W_0, [144, 144])
    fc3_W_0 = fc3_W_0.T
    final_fc_0 = np.reshape(final_fc_0, [12, 144])
    final_fc_0 = final_fc_0.T

    sess = tf.Session()
    flow = tf.constant(flow)
    fc1_W_0 = tf.constant(fc1_W_0)
    fc1_b_0 = tf.constant(fc1_b_0)
    sess.run(tf.Print(fc1_W_0, [fc1_W_0], summarize=10))
    flow = tf.matmul(flow, fc1_W_0) + fc1_b_0
    flow = tf.nn.relu(flow)
    sess.run(tf.Print(flow, [flow], summarize=10))
    fc2_W_0 = tf.constant(fc2_W_0)
    fc2_b_0 = tf.constant(fc2_b_0)
    flow = tf.matmul(flow, fc2_W_0) + fc2_b_0
    flow = tf.nn.relu(flow)
    sess.run(tf.Print(flow, [flow], summarize=10))
    fc3_W_0 = tf.constant(fc3_W_0)
    fc3_b_0 = tf.constant(fc3_b_0)
    flow = tf.matmul(flow, fc3_W_0) + fc3_b_0
    flow = tf.nn.relu(flow)
    sess.run(tf.Print(flow, [flow], summarize=10))
    final_fc_0 = tf.constant(final_fc_0)
    Variable_0 = tf.constant(Variable_0)
    flow = tf.matmul(flow, final_fc_0) + Variable_0
    sess.run(tf.Print(flow, [flow], summarize=10))
    flow = tf.nn.softmax(flow, name='labels_softmax')
    predictions = sess.run(tf.Print(flow, [flow], summarize=10))
    predictions = predictions[0]
    top_k = predictions.argsort()[-3:][::-1]
    labels = ["Silence", "Unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))

    # flow = np.dot(flow, fc1_W_0) + fc1_b_0
    # flow = np.maximum(flow, 0)
    # flow = np.dot(flow, fc2_W_0) + fc2_b_0
    # flow = np.maximum(flow, 0)
    # flow = np.dot(flow, fc3_W_0) + fc3_b_0
    # flow = np.maximum(flow, 0)
    # logits = np.dot(flow, final_fc_0) + Variable_0
    # index = np.argmax(logits, 1)[0]
    # rsMap = ["Silence", "Unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    # print(logits, index, rsMap[index])


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    tf.app.run(main=main)
