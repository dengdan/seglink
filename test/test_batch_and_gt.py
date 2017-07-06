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
DATA_FORMAT = 'NHWC'

# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 18,
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
    'model_name', 'seglink_vgg', 'The name of the architecture to train.')
tf.app.flags.DEFINE_integer(
    'batch_size', 2, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('train_image_width', 1024, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_height', 512, 'Train image size')


FLAGS = tf.app.flags.FLAGS
    
    
def draw_bbox(mask, bbox, color = util.img.COLOR_RGB_RED):
    bbox = np.reshape(bbox, (4, 2))
    cnts = util.img.points_to_contours(bbox)
    util.img.draw_contours(mask, cnts, -1, color = color)
    
def config_initialization():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    
    # image shape and feature layers shape inference
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)
    
    config.init_config(image_shape, batch_size = FLAGS.batch_size)

    util.proc.set_proc_name(FLAGS.model_name + '_' + FLAGS.dataset_name)
    
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
#     config.print_config(FLAGS, dataset)
    return dataset
    
def create_dataset_batch_queue(dataset):
    batch_size = config.batch_size
    with tf.device('/cpu:0'):
        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=20 * batch_size,
                common_queue_min=10 * batch_size,
                shuffle=True)
        # Get for SSD network: image, labels, bboxes.
        [image, shape, gignored, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get([
                                                         'image', 'shape',
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
        image, gignored, gbboxes, gxs, gys = ssd_vgg_preprocessing.preprocess_image(
                                                           image, gignored, gbboxes, gxs, gys, 
                                                           out_shape = config.image_shape,
                                                           data_format = config.data_format, 
                                                           is_training = True)
        image = tf.identity(image, 'processed_image')
        
        # calculate ground truth
        seg_label, seg_offsets, link_label = seglink.tf_get_all_seglink_gt(gxs, gys, gignored)

        # batch them
        b_image, b_seg_label, b_seg_offsets, b_link_label = tf.train.batch(
            [image, seg_label, seg_offsets, link_label],
            batch_size = config.batch_size_per_gpu,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity = 50)
            
        batch_queue = slim.prefetch_queue.prefetch_queue(
            [b_image, b_seg_label, b_seg_offsets, b_link_label],
            capacity = 50) 
    return batch_queue    

# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    util.init_logger()
    dump_path = util.io.get_absolute_path('~/temp/no-use/seglink/')
    
    dataset = config_initialization()
    batch_queue = create_dataset_batch_queue(dataset)
    batch_size = config.batch_size
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        b_image, b_seg_label, b_seg_offsets, b_link_label = batch_queue.dequeue()
        batch_idx = 0;
        while True: #batch_idx < 50:
            image_data_batch, seg_label_data_batch, seg_offsets_data_batch, link_label_data_batch = \
                            sess.run([b_image, b_seg_label, b_seg_offsets, b_link_label])
            for image_idx in xrange(batch_size):
                image_data = image_data_batch[image_idx, ...]
                seg_label_data = seg_label_data_batch[image_idx, ...]
                seg_offsets_data = seg_offsets_data_batch[image_idx, ...]
                link_label_data = link_label_data_batch[image_idx, ...]
                
                image_data = image_data + [123, 117, 104]
                image_data = np.asarray(image_data, dtype = np.uint8)
                
                # decode the encoded ground truth back to bboxes
                bboxes = seglink.seglink_to_bbox(seg_scores = seg_label_data, 
                                                 link_scores = link_label_data, 
                                                 seg_offsets_pred = seg_offsets_data)
                
                # draw bboxes on the image
                for bbox_idx in xrange(len(bboxes)):
                    bbox = bboxes[bbox_idx, :] 
                    draw_bbox(image_data, bbox)
                
                image_path = util.io.join_path(dump_path, '%d_%d.jpg'%(batch_idx, image_idx))
                util.plt.imwrite(image_path, image_data)
                print 'Make sure that the text on the image are correctly bounded\
                                                         with oriented boxes:', image_path 
            batch_idx += 1
                
                
if __name__ == '__main__':
    tf.app.run()
