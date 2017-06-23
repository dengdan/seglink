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
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')


FLAGS = tf.app.flags.FLAGS
    
    
def draw_oriented_rect(mask, rect, text_pos = None, color = util.img.COLOR_RGB_RED):
    if text_pos:
        util.img.put_text(mask, pos = text_pos, scale=0.5, text = 'cv2: cx=%d, cy=%d, w=%d, h=%d, theta_0=%f'%(rect[0], rect[1], rect[2], rect[3], rect[4]))
    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    box_center = rect[0]
    util.img.circle(mask, box_center, 3, color = color)
    cv2.drawContours(mask, [box], 0, color, 1)
    
    
def config_initialization():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    
    # image shape and feature layers shape inference
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)
    

    config.init_config(image_shape, batch_size = 8)

    batch_size = config.batch_size
    batch_size_per_gpu = config.batch_size_per_gpu
        
    tf.summary.scalar('batch_size', batch_size)
    tf.summary.scalar('batch_size_per_gpu', batch_size_per_gpu)
  
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
        image, glabels, gbboxes, gxs, gys = ssd_vgg_preprocessing.preprocess_image(image, glabels, gbboxes, gxs, gys, 
                                                           out_shape = config.image_shape,
                                                           data_format = config.data_format, 
                                                           is_training = True)
        image = tf.identity(image, 'processed_image')
        
        # calculate ground truth
        seg_label, seg_loc, link_gt = seglink.tf_get_all_seglink_gt(gxs, gys)
        
        # batch them
        b_image, b_seg_label, b_seg_loc, b_link_gt = tf.train.batch(
            [image, seg_label, seg_loc, link_gt],
            batch_size = config.batch_size_per_gpu,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity = 50)
            
        batch_queue = slim.prefetch_queue.prefetch_queue(
            [b_image, b_seg_label, b_seg_loc, b_link_gt],
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
        b_image, b_seg_labels, b_seg_gt, b_link_gt = batch_queue.dequeue()
        batch_idx = 0;
        train_writer = tf.summary.FileWriter(dump_path, sess.graph)
        
        while True: #batch_idx < 50:
            @util.dec.print_calling_in_short # print time consumed
            def run():
                return sess.run([b_image, b_seg_labels, b_seg_gt, b_link_gt, summary_op])
            image_data_batch, seg_label_data_batch, seg_gt_data_batch, link_gt_data_batch, summary_op_batch = run()
            for image_idx in xrange(batch_size):
                train_writer.add_summary(summary_op_batch, batch_idx)
                image_data = image_data_batch[image_idx, ...]
                seg_label_data = seg_label_data_batch[image_idx, ...]
                seg_gt_data = seg_gt_data_batch[image_idx, ...]
                link_gt_data = link_gt_data_batch[image_idx, ...]
                
                
                image_data = image_data + [123, 117, 104]
                image_data = np.asarray(image_data, dtype = np.uint8)
                
                h_I, w_I = config.image_shape
                
                # decode the encoded ground truth back to bboxes
                bboxes = seglink.seglink_to_bbox(seg_scores = seg_label_data, link_scores = link_gt_data, seg_offsets_pred = seg_gt_data)
                
                # draw the bboxes on the image
                for bbox_idx in xrange(len(bboxes)):
                    bbox = bboxes[bbox_idx, :] 
                    draw_oriented_rect(image_data, bbox)
                
                image_path = util.io.join_path(dump_path, '%d_%d.jpg'%(batch_idx, image_idx))
                util.plt.imwrite(image_path, image_data)
                print 'Make sure that the text on the image are correctly bounding with oriented boxes:', image_path 
            batch_idx += 1
                
                
if __name__ == '__main__':
    tf.app.run()
