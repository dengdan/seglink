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
from tf_extended import seglink as tfe_seglink
import util
slim = tf.contrib.slim


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
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', '~/dataset/SSD-tf/SynthText', 'The directory where the dataset files are stored.')
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
    batch_size = FLAGS.batch_size;
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
            [image, shape, gignored, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get(['image', 'shape',
                                                             'object/ignored',
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
            image_shape = (FLAGS.train_image_size, FLAGS.train_image_size)
            image, gignored, gbboxes, gxs, gys = \
                            ssd_vgg_preprocessing.preprocess_image(image, gignored, gbboxes, gxs, gys, 
                                                               out_shape=image_shape,
                                                               is_training = True)
            gxs = gxs * tf.cast(image_shape[1], gxs.dtype)
            gys = gys * tf.cast(image_shape[0], gys.dtype)
            gorbboxes = tfe_seglink.tf_min_area_rect(gxs, gys)
            image = tf.identity(image, 'processed_image')
            
            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                i = 0
                while i < 2:
                    i += 1
                    image_data, label_data, bbox_data, xs_data, ys_data, orbboxes = \
                                 sess.run([image, gignored, gbboxes, gxs, gys, gorbboxes])
                    image_data = image_data + [123., 117., 104.]
                    image_data = np.asarray(image_data, np.uint8)
                    h, w = image_data.shape[0:-1]
                    bbox_data = bbox_data * [h, w, h, w]
                    I_bbox = image_data.copy()
                    I_xys = image_data.copy()
                    I_orbbox = image_data.copy()
                    
                    for idx in range(bbox_data.shape[0]):
                        
                        def draw_bbox():
                            y1, x1, y2, x2 = bbox_data[idx, :]
                            util.img.rectangle(I_bbox, (x1, y1), (x2, y2), color = util.img.COLOR_WHITE)
                        
                        def draw_xys():
                            points = zip(xs_data[idx, :], ys_data[idx, :])
                            cnts = util.img.points_to_contours(points);
                            util.img.draw_contours(I_xys, cnts, -1, color = util.img.COLOR_GREEN)

                        def draw_orbbox():
                            orbox = orbboxes[idx, :]
                            import cv2
                            rect = ((orbox[0], orbox[1]), (orbox[2], orbox[3]), orbox[4])
                            box = cv2.cv.BoxPoints(rect)
                            box = np.int0(box)
                            cv2.drawContours(I_orbbox, [box], 0, util.img.COLOR_RGB_RED, 1)
                        
                        draw_bbox()
                        draw_xys();
                        draw_orbbox();
                        
                    print util.sit(I_bbox)
                    print util.sit(I_xys)
                    print util.sit(I_orbbox)
                    print 'check the images and make sure that bboxes in difference colors are the same.'
                coord.request_stop()
                coord.join(threads)
if __name__ == '__main__':
    tf.app.run()
