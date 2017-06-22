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
tf.app.flags.DEFINE_integer('train_image_width', 1024, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_height', 512, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')


FLAGS = tf.app.flags.FLAGS
    
def draw_horizontal_rect(mask, rect, text_pos = None, color = util.img.COLOR_GREEN, draw_center = True, center_only = False):
    if text_pos is not None:
        if len(rect) == 5:
            util.img.put_text(mask, pos = text_pos, scale=0.5, text = 'trans: cx=%d, cy=%d, w=%d, h=%d, theta_0=%f'%(rect[0], rect[1], rect[2], rect[3], rect[4]))
        else:
            util.img.put_text(mask, pos = text_pos, scale=0.5, text = 'trans: cx=%d, cy=%d, w=%d, h=%d, theta_0=0.0'%(rect[0], rect[1], rect[2], rect[3]))
    rect = np.asarray(rect, dtype = np.float32)
    cx, cy, w, h = rect[0:4]
    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2
    if draw_center or center_only:
        util.img.circle(mask, (cx, cy), 3, color = color)
    
    if not center_only:
        util.img.rectangle(mask, (xmin, ymin), (xmax, ymax), color = color)
    
    
    
def draw_oriented_rect(mask, rect, text_pos = None, color = util.img.COLOR_RGB_RED):
    if text_pos:
        util.img.put_text(mask, pos = text_pos, scale=0.5, text = 'cv2: cx=%d, cy=%d, w=%d, h=%d, theta_0=%f'%(rect[0], rect[1], rect[2], rect[3], rect[4]))
    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    box_center = rect[0]
    util.img.circle(mask, box_center, 3, color = color)
    cv2.drawContours(mask, [box], 0, color, 1)
    
    
def points_to_xys(points):
    points = np.asarray(points, dtype = np.float32)
    points = np.reshape(points, (-1, 4, 2))
    xs = points[..., 0]
    ys = points[..., 1]
    return xs, ys

def config_initialization():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    
    # image shape and feature layers shape inference
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)
    

    config.init_config(image_shape, batch_size = FLAGS.batch_size, weight_decay = 0.01, num_gpus = 1)

    batch_size = config.batch_size
    batch_size_per_gpu = config.batch_size_per_gpu
        
    tf.summary.scalar('batch_size', batch_size)
    tf.summary.scalar('batch_size_per_gpu', batch_size_per_gpu)

    util.proc.set_proc_name(FLAGS.model_name + '_' + FLAGS.dataset_name)
    
    
def create_dataset_batch_queue():
    batch_size = config.batch_size
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
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
    config_initialization()
    batch_queue = create_dataset_batch_queue()
    batch_size = config.batch_size
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        tf.train.start_queue_runners(sess)
        b_image, b_seg_labels, b_seg_gt, b_link_gt = batch_queue.dequeue()
        batch_idx = 0;
        train_writer = tf.summary.FileWriter(util.io.get_absolute_path('~/temp/no-use/seglink/'), sess.graph)
        
        while True:
            image_datas, label_datas, seg_datas, link_datas, so = sess.run([b_image, b_seg_labels, b_seg_gt, b_link_gt, summary_op])
            for image_idx in xrange(batch_size):
                train_writer.add_summary(so, batch_idx)
                image_data = image_datas[image_idx, ...]
                label_data = label_datas[image_idx, ...]
                seg_data = seg_datas[image_idx, ...]
                link_data = link_datas[image_idx, ...]
                
                image_data = image_data + [123, 117, 104]
                image_data = np.asarray(image_data, dtype = np.uint8)
                h_I, w_I = config.image_shape
                bboxes = seglink.seglink_to_bbox(seg_scores = label_data, link_scores = link_data, segs = seg_data)
                if len(bboxes) == 0:
                    util.plt.imwrite('~/temp/no-use/seglink/no-bboxes/%d_%d.jpg'%(batch_idx, image_idx), image_data)
                    print "no bboxes on the image"
                    continue
                
                bboxes = bboxes * [w_I, h_I, w_I, h_I, 1]
                seg_data = seg_data * [w_I, h_I, w_I, h_I, 1]
                seg_groups = seglink.group_segs(seg_scores = label_data, link_scores = link_data)
                img = image_data.copy()
                for group, bbox in zip(seg_groups, bboxes):
                    for seg_idx in group:
                        seg = seg_data[seg_idx, :]
                        #draw_oriented_rect(img, seg, color = util.img.COLOR_RGB_YELLOW)
                    draw_oriented_rect(img, bbox, color = util.img.COLOR_GREEN)
                    #draw_line(img, bbox[-2], bbox[-1], color = util.img.COLOR_RGB_RED)
            util.plt.imwrite('~/temp/no-use/seglink/%d_%d.jpg'%(batch_idx, image_idx), img)
            print 'batch: %d'%(batch_idx)
            batch_idx += 1
                
                
if __name__ == '__main__':
    tf.app.run()
