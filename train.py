# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modifications Copyright 2017 Arm Inc. All Rights Reserved.           
# Added model dimensions as command line argument and changed to Adam optimizer
#
#
"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /Users/zoutai/ML_KWS/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os.path
import sys

import numpy as np
import tensorflow as tf
from six.moves import range  # pylint: disable=redefined-builtin
from tensorflow.contrib import slim as slim
from tensorflow.python.platform import gfile

import input_data
import models

FLAGS = None


def main(_):
    # We want to see all the logging messages for this tutorial.
    # 记录日志
    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    # 启动会话
    sess = tf.InteractiveSession()

    # 将日志写入指定文件
    # get TF logger
    import utils
    utils.create_log_dir(FLAGS.train_dir,FLAGS.log_path);

    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    # 模型设置：单词切割、采样率、时长、窗长、帧移、mfcc系数
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)

    # 数据网址、本地文件夹、静音比重、干扰词比重、验证集占比、测试集占比
    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings)

    # 数据输入格式
    fingerprint_size = model_settings['fingerprint_size']
    # 标签数量
    label_count = model_settings['label_count']
    # 采样率
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

    # Figure out the learning rates for each training phase. Since it's often
    # effective to have high learning rates at the start of training, followed by
    # lower levels towards the end, the number of steps and learning rates can be
    # specified as comma-separated lists to define the rate at each stage. For
    # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
    # will run 13,000 training loops in total, with a rate of 0.001 for the first
    # 10,000, and 0.0001 for the final 3,000.
    # 实验发现：在训练的开始，能很快到达很高的准确率，即训练速度快；但在接近最高点的时候，学习效率就会降低；
    # 因此为了适应前后训练的不同，需要根据时间调整前后的学习率：前面大，后面小；
    # 比如整个循环有13000次，前10000次使用0.001的学习率；后3000次使用0.0001的学习率
    # training_steps_list=10000,10000,10000
    # learning_rates_list=0.0005, 0.0001, 0.00002
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                       len(learning_rates_list)))

    # 数据输入占位符
    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    # 创建模型
    # logits:预测标签结果;dropout_prob:丢弃率
    logits, dropout_prob = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        FLAGS.model_size_info,
        is_training=True)

    # Define loss and optimizer
    # 输出维度占位符---真实标签
    ground_truth_input = tf.placeholder(
        tf.float32, [None, label_count], name='groundtruth_input')

    # Optionally we can add runtime checks to spot when NaNs or other symptoms of
    # numerical errors start occurring during training.
    control_dependencies = []
    if FLAGS.check_nans:
        checks = tf.add_check_numerics_ops()
        control_dependencies = [checks]

    # Create the back propagation and training evaluation machinery in the graph.
    # 交叉验证，即优化函数
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=ground_truth_input, logits=logits))
    # summary概要：主要用于数据可视化，scalar将数据转化为标量
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    # 使用BN，即批量更新
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope('train'), tf.control_dependencies(update_ops), tf.control_dependencies(control_dependencies):
        # 学习率
        learning_rate_input = tf.placeholder(
            tf.float32, [], name='learning_rate_input')
        # 优化函数
        train_op = tf.train.AdamOptimizer(
            learning_rate_input)
        # 交叉验证函数+优化函数 进行训练
        train_step = slim.learning.create_train_op(cross_entropy_mean, train_op)

    #    train_step = tf.train.GradientDescentOptimizer(
    #        learning_rate_input).minimize(cross_entropy_mean)

    # 返回概率最大的标签下标：分别为预测值-真实值
    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    # 正确与否，0-1值矩阵
    correct_prediction = tf.equal(predicted_indices, expected_indices)

    # 计算混淆矩阵：直接显示模型的效果，越集中于对角线，表示效果越好
    # https: // www.zhihu.com / question / 36883196
    confusion_matrix = tf.confusion_matrix(
        expected_indices, predicted_indices, num_classes=label_count)

    # 正确率：计算平均值，因为由0-1组成，所以平均值即正确率
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    # 全局步数记录
    global_step = tf.train.get_or_create_global_step()
    # 全局步数+1
    increment_global_step = tf.assign(global_step, global_step + 1)

    # 模型存储
    saver = tf.train.Saver(tf.global_variables())

    # Merge all the summaries and write them out to /Users/zoutai/ML_KWS/retrain_logs (by default)
    # 将所有的数据概要保存到文件
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    # 初始化全局变量
    tf.global_variables_initializer().run()

    # Parameter counts
    # 记录参数量
    params = tf.trainable_variables()
    num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
    # print('Total number of Parameters: ', num_params)
    tf.logging.info('Total number of Parameters: %d ', num_params)

    start_step = 1

    # 是否有初始模型，如果有从初始模型开始；否则，重新开始训练
    if FLAGS.start_checkpoint:
        models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=sess)

    start_step = start_step // 2
    tf.logging.info('Training from step: %d ', start_step)

    # Save graph.pbtxt.
    # 存储模型
    tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                         FLAGS.model_architecture + '.pbtxt')

    # Save list of words.
    # 记录标签
    with gfile.GFile(
            os.path.join(FLAGS.train_dir, FLAGS.model_architecture + '_labels.txt'),
            'w') as f:
        f.write('\n'.join(audio_processor.words_list))

    # 开启训练
    # Training loop.
    best_accuracy = 0
    # 总训练步数
    training_steps_max = np.sum(training_steps_list)
    for training_step in range(start_step, training_steps_max + 1):
        # Figure out what the current learning rate is.
        # 每一阶段的总步数
        training_steps_sum = 0

        # 三个阶段
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate_value = learning_rates_list[i]
                break

        # Pull the audio samples we'll use for training.
        # 读取输入数据
        train_fingerprints, train_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
            FLAGS.background_volume, time_shift_samples, 'training', sess)

        # 1.训练部分
        # Run the graph with this batch of training data.
        # 开启训练会话
        train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
            [
                merged_summaries, evaluation_step, cross_entropy_mean, train_step,
                increment_global_step
            ],
            feed_dict={
                fingerprint_input: train_fingerprints,
                ground_truth_input: train_ground_truth,
                learning_rate_input: learning_rate_value,
                dropout_prob: 1.0
            })

        train_writer.add_summary(train_summary, training_step)
        tf.logging.info('Step #%d: rate %f, accuracy %.2f%%, cross entropy %f' %
                        (training_step, learning_rate_value, train_accuracy * 100,
                         cross_entropy_value))

        # 2.验证部分
        # 最后一步，训练完成
        is_last_step = (training_step == training_steps_max)
        # 达到一定步数进行一次记录，一般为500次；或者到达最终步数记录一次
        if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
            # 读取验证集，进行验证
            set_size = audio_processor.set_size('validation')
            total_accuracy = 0
            total_conf_matrix = None
            for i in range(0, set_size, FLAGS.batch_size):
                validation_fingerprints, validation_ground_truth = (
                    audio_processor.get_data(FLAGS.batch_size, i, model_settings, 0.0,
                                             0.0, 0, 'validation', sess))

                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy, conf_matrix = sess.run(
                    [merged_summaries, evaluation_step, confusion_matrix],
                    feed_dict={
                        fingerprint_input: validation_fingerprints,
                        ground_truth_input: validation_ground_truth,
                        dropout_prob: 1.0
                    })
                validation_writer.add_summary(validation_summary, training_step)
                batch_size = min(FLAGS.batch_size, set_size - i)
                total_accuracy += (validation_accuracy * batch_size) / set_size
                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    total_conf_matrix += conf_matrix
            tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
            tf.logging.info('Step %d: Validation accuracy = %.2f%% (N=%d)' %
                            (training_step, total_accuracy * 100, set_size))

            # Save the model checkpoint when validation accuracy improves
            # 存储当前模型
            if total_accuracy > best_accuracy:
                best_accuracy = total_accuracy
                checkpoint_path = os.path.join(FLAGS.train_dir, 'best',
                                               FLAGS.model_architecture + '_' + str(
                                                   int(best_accuracy * 10000)) + '.ckpt')
                tf.logging.info('Saving best model to "%s-%d"', checkpoint_path, training_step)
                saver.save(sess, checkpoint_path, global_step=training_step)
            tf.logging.info('So far the best validation accuracy is %.2f%%' % (best_accuracy * 100))

    # 3.测试部分
    set_size = audio_processor.set_size('testing')
    tf.logging.info('set_size=%d', set_size)
    total_accuracy = 0
    total_conf_matrix = None
    for i in range(0, set_size, FLAGS.batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
        test_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                fingerprint_input: test_fingerprints,
                ground_truth_input: test_ground_truth,
                dropout_prob: 1.0
            })
        batch_size = min(FLAGS.batch_size, set_size - i)
        total_accuracy += (test_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.logging.info('Final test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                             set_size))

    # 验证集和测试集的区别：
    # 验证集用于模型训练；测试集不用于训练，只用于测试最终的模型效果

if __name__ == '__main__':
    import os
    # 是否使用GPU加速，-1代表仅使用CPU，0、1、2分别表示GPU编号
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    # mac报错处理：OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    parser = argparse.ArgumentParser()
    # 数据下载地址
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    # 数据本地地址
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Users/zoutai/ML_KWS/King-ASR-M-005-new/',
        help="""\
      Where to download the speech training data to.
      """)
    # 背景噪音音量
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.1,
        help="""\
      How loud the background noise should be, between 0 and 1.
      """)
    # 多少样本需要掺杂噪音
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
      How many of the training samples have background noise mixed in.
      """)
    # 多少样本需要静音
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be silence.
      """)
    # 过滤词占比：即非关键词
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be unknown words.
      """)
    # 帧移
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
      Range to randomly shift the training audio by in time.
      """)
    # 测试集比重
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    # 验证集比重
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    # 语音采样率：每一秒采样数，一般为16000
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    # 输入音频时长ms
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs', )
    # 窗长：帧
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is', )
    # 帧移
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long each spectrogram timeslice is', )
    # MFCC系数
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint', )
    # 分阶段训练步数
    parser.add_argument(
        '--how_many_training_steps',
        type=str,
        default='20000,20000,20000',
        help='How many training loops to run', )
    # 多少步评估一次
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=400,
        help='How often to evaluate the training results.')
    # 分阶段学习率
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.0005,0.0001,0.00002',
        help='How large a learning rate to use when training.')
    # 一个批次训练多少样本
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once', )
    # 概要日志保存地址
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/Users/zoutai/ML_KWS/retrain_logs',
        help='Where to save summary logs for TensorBoard.')
    # 关键词标签
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)', )
    # 日志和模型保存文件夹
    parser.add_argument(
        '--train_dir',
        type=str,
        default='',
        help='Directory to write event logs and checkpoint.')
    # 多少步保存一次模型
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=400,
        help='Save model checkpoint every save_steps.')
    # 是否有初始模型
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    # 选择模型
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='cnn_attention',
        help='What model architecture to use')
    # 模型对应的模型参数
    parser.add_argument(
        '--model_size_info',
        type=int,
        nargs="+",
        default=[98, 144],
        help='Model dimensions - different for various models')
    # 是否检查无效的数字
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')
    # 日志存储位置(新增)
    parser.add_argument(
        '--log_path',
        type=str,
        default='tensorflow.log',
        help='Log storage location')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
