"""
@author: zoutai
@file: testRnn.py
@time: 2018/11/19
@description: 测试audio_ops是否是随机截断
"""
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
import tensorflow as tf

wav_loader = io_ops.read_file('/home/zoutai/DataSets/speech_dataset/follow/54aecbd5_nohash_0.wav')
wav_decoder0 = contrib_audio.decode_wav(
    wav_loader, desired_channels=1, desired_samples=10000)

wav_decoder1 = contrib_audio.decode_wav(
    wav_loader, desired_channels=1, desired_samples=10000)

aeqb = tf.equal(wav_decoder0.audio, wav_decoder1.audio)
aeqb_int = tf.to_int32(aeqb)
result = tf.equal(tf.reduce_sum(aeqb_int), tf.reduce_sum(tf.ones_like(aeqb_int)))

# Launch the graph in a session.
sess = tf.Session()
# Evaluate the tensor `c`.
print(sess.run(result))
