# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""test code to make sure the preprocessing works all right"""
import numpy as np
import tensorflow as tf

from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from tf_extended import seglink
import util
from nets import seglink_symbol
from nets import anchor_layer
slim = tf.contrib.slim
import config
DATA_FORMAT = 'NHWC'

# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'synthtext', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', util.io.get_absolute_path('~/dataset/SSD-tf/SynthText'), 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', 'ssd_vgg', 'The name of the architecture to train.')
tf.app.flags.DEFINE_integer(
    'batch_size', 2, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', 512, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')


FLAGS = tf.app.flags.FLAGS

# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.DEBUG)

    batch_size = FLAGS.batch_size
    image_shape = (FLAGS.train_image_size, FLAGS.train_image_size)
    config.init_config(image_shape)
    
    with tf.Graph().as_default():
        # Select the dataset.
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        util.proc.set_proc_name(FLAGS.model_name + '_' + FLAGS.dataset_name)


        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.device('/cpu:0'):
            with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20 * batch_size,
                    common_queue_min=10 * batch_size,
                    shuffle=True)
            # Get for SSD network: image, labels, bboxes.
            [image, shape, glabels, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get(['image', 'shape',
                                                             'object/label',
                                                             'object/bbox', 
                                                             'object/oriented_bbox/x1',
                                                             'object/oriented_bbox/x2',
                                                             'object/oriented_bbox/x3',
                                                             'object/oriented_bbox/x4',
                                                             'object/oriented_bbox/y1',
                                                             'object/oriented_bbox/y2',
                                                             'object/oriented_bbox/y3',
                                                             'object/oriented_bbox/y4'
                                                             ])
            gxs = tf.transpose(tf.stack([x1, x2, x3, x4])) #shape = (N, 4)
            gys = tf.transpose(tf.stack([y1, y2, y3, y4]))
            image = tf.identity(image, 'input_image')
            # Pre-processing image, labels and bboxes.

            image, glabels, gbboxes, gxs, gys = \
                            ssd_vgg_preprocessing.preprocess_image(image, glabels, gbboxes, gxs, gys, 
                                                               out_shape=image_shape,
                                                               data_format=DATA_FORMAT,
                                                               is_training = True)
            image = tf.identity(image, 'processed_image')
            
            
            # calculate ground truth
            seg_labels, seg_gt, link_gt = seglink.tf_get_all_seglink_gt(gxs, gys)
            
            # batch them
            b_image, b_seg_labels, b_seg_gt, b_link_gt = tf.train.batch(
                [image, seg_labels, seg_gt, link_gt],
                batch_size = batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity = 50 * batch_size)
                
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [b_image, b_seg_labels, b_seg_gt, b_link_gt],
                capacity = 2) 

            with tf.Session() as sess:
                tf.train.start_queue_runners(sess)
                b_image, b_seg_labels, b_seg_gt, b_link_gt = batch_queue.dequeue()
                while True:
                    import pdb
                    pdb.set_trace()
                    image_data, label_data, seg_data, link_data = sess.run([b_image, b_seg_labels, b_seg_gt, b_link_gt])                
                
                
                
                
if __name__ == '__main__':
    tf.app.run()
