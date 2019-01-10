""" 
@author: zoutai
@file: testOnewave.py
@time: 2018/12/19 
@description: test.py的复制版：用于测试单个语音，便于调试
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import tensorflow as tf
import input_data
import models
import numpy as np


def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


def run_inference(labels, wanted_words, sample_rate, clip_duration_ms,
                  window_size_ms, window_stride_ms, dct_coefficient_count,):
    """Creates an audio model with the nodes needed for inference.

    Uses the supplied arguments to create a model, and inserts the input and
    output nodes that are needed to use the graph for inference.

    Args:
      wanted_words: Comma-separated list of the words we're trying to recognize.
      sample_rate: How many samples per second are in the input audio files.
      clip_duration_ms: How many samples to analyze for the audio pattern.
      window_size_ms: Time slice duration to estimate frequencies from.
      window_stride_ms: How far apart time slices should be.
      dct_coefficient_count: Number of frequency bands to analyze.
      model_architecture: Name of the kind of model to generate.
      model_size_info: Model dimensions : different lengths for different models
    """

    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()
    words_list = input_data.prepare_words_list(wanted_words.split(','))
    model_settings = models.prepare_model_settings(
        len(words_list), sample_rate, clip_duration_ms, window_size_ms,
        window_stride_ms, dct_coefficient_count)

    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings)

    fingerprint_size = model_settings['fingerprint_size']

    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits,arr = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        FLAGS.model_size_info,
        is_training=False)

    models.load_variables_from_checkpoint(sess, FLAGS.checkpoint)

    training_fingerprints = audio_processor.get_onedata(sess,filename=FLAGS.data_dir)
    print(training_fingerprints)
    logits,arr = sess.run(
        [logits,arr],
        feed_dict={
            fingerprint_input: training_fingerprints,
        })
    num_top_predictions=3

    # dnn测试用
    predictions = logits[0]
    # predictions = logits
    predictions = softmax(predictions)
    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]

    labels = load_labels(labels)
    for node_id in top_k:
        human_string = labels[node_id]
        score = predictions[node_id]
        print('%s (score = %.5f)' % (human_string, score))


def main(_):
    # Create the model, load weights from checkpoint and run on train/val/test
    run_inference(FLAGS.labels, FLAGS.wanted_words, FLAGS.sample_rate,
                  FLAGS.clip_duration_ms, FLAGS.window_size_ms,
                  FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--labels', type=str, default='/home/zoutai/code/ML-KWS-for-MCU/Pretrained_models/labels.txt',
        help='Path to file containing labels.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/zoutai/DataSets/speech_data/yes/ffd2ba2f_nohash_4.wav',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be silence.
      """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be unknown words.
      """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=100,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=0,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=40.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=40.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=10,
        help='How many bins to use for the MFCC fingerprint', )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='How many items to train with at once', )
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/zoutai/code/ML-KWS-for-MCU/work/DS_CNN/DS_CNN1/training/best/ds_cnn_9347.ckpt-26400',
        help='Checkpoint to load the weights from.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='ds_cnn',
        help='What model architecture to use')
    parser.add_argument(
        '--model_size_info',
        type=int,
        nargs="+",
        default=[5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1],
        help='Model dimensions - different for various models')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
