#test code to make sure the ground truth calculation and data batch works well.

import numpy as np
import tensorflow as tf # test

from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from tf_extended import seglink
import util
import cv2
from nets import seglink_symbol
from nets import anchor_layer
slim = tf.contrib.slim
import config

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
tf.app.flags.DEFINE_integer('train_image_width', 512, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_height', 1024, 'Train image size')
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
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)
    config.init_config(image_shape)
    
    with tf.Graph().as_default():
        # Select the dataset.
        dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

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
                                                               data_format = config.data_format, 
                                                               is_training = True)
            image = tf.identity(image, 'processed_image')
            
            
            # calculate ground truth
            seg_label, seg_loc, link_gt = seglink.tf_get_all_seglink_gt(gxs, gys)
            
            # batch them
            b_image, b_seg_label, b_seg_loc, b_link_gt = tf.train.batch(
                [image, seg_label, seg_loc, link_gt],
                batch_size = batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity = 50 * batch_size)
                
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [b_image, b_seg_label, b_seg_loc, b_link_gt],
                capacity = 2) 

            def clone_fn():
                b_image, b_seg_label, b_seg_loc, b_link_gt = batch_queue.dequeue()
                with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)):
                    net = seglink_symbol.SegLinkNet(inputs = b_image, data_format = config.data_format)
                    loss = net.build_loss(seg_label = b_seg_label, seg_loc = b_seg_loc, link_gt = b_link_gt,
                                          seg_loc_loss_weight = 1.0, link_conf_loss_weight = 1.0)
                    
if __name__ == '__main__':
    tf.app.run()
