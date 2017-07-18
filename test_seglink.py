#encoding = utf-8
"""Read test images, and store the detection result as txt files and zip file. 
    The zip file follows the rule of ICDAR2015 Challenge4 Task1
"""
import numpy as np
import math
import tensorflow as tf # test
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.training.python.training import evaluation
from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from tf_extended import seglink, metrics
import util
import cv2
from nets import seglink_symbol, anchor_layer

slim = tf.contrib.slim
import config
# =========================================================================== #
# model threshold parameters
# =========================================================================== #
tf.app.flags.DEFINE_float('seg_conf_threshold', 0.9, 
                          'the threshold on the confidence of segment')
tf.app.flags.DEFINE_float('link_conf_threshold', 0.7, 
                          'the threshold on the confidence of linkage')

# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of checkpoint to be evaluated. If it is a directory containing many checkpoints, the lastest will be evaluated.')
tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, 'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')


# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'icdar2015', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')
tf.app.flags.DEFINE_string('dataset_dir', 
           util.io.get_absolute_path('~/dataset/ICDAR2015/Challenge4/ch4_test_images'), 
           'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', 'seglink_vgg', 'The name of the architecture to train.')
tf.app.flags.DEFINE_integer('eval_image_width', 1280, 'Train image size')
tf.app.flags.DEFINE_integer('eval_image_height', 768, 'Train image size')


FLAGS = tf.app.flags.FLAGS

def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.eval_image_height, FLAGS.eval_image_width)
    
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    
    config.init_config(image_shape, batch_size = 1, seg_conf_threshold = FLAGS.seg_conf_threshold,
                       link_conf_threshold = FLAGS.link_conf_threshold)

    util.proc.set_proc_name('test' + FLAGS.model_name)

    
def write_result(image_name, image_data, bboxes, path):
  filename = util.io.join_path(path, 'res_%s.txt'%(image_name))
  print filename
  lines = []
  for bbox in bboxes:
        line = "%d, %d, %d, %d, %d, %d, %d, %d\r\n"%(int(v) for v in bbox)
        lines.append(line)
  util.io.write_lines(filename, lines)

  
def eval():
    
    with tf.name_scope('test'):
        with tf.variable_scope(tf.get_variable_scope(), reuse = True):# the variables has been created in config.init_config
            image = tf.placeholder(dtype=tf.int32, shape = [None, None, 3])
            image_shape = tf.placeholder(dtype = tf.int32, shape = [3, ])
            processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(image, None, None, None, None, 
                                                       out_shape = config.image_shape,
                                                       data_format = config.data_format, 
                                                       is_training = False)
            b_image = tf.expand_dims(processed_image, axis = 0)
            b_shape = tf.expand_dims(image_shape, axis = 0)
            net = seglink_symbol.SegLinkNet(inputs = b_image, data_format = config.data_format)
            bboxes_pred = seglink.tf_seglink_to_bbox(net.seg_scores, net.link_scores, 
                                                     net.seg_offsets, 
                                                     image_shape = b_shape, 
                                                     seg_conf_threshold = config.seg_conf_threshold,
                                                     link_conf_threshold = config.link_conf_threshold)

    image_names = util.io.ls(FLAGS.dataset_dir)
    
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;
    
    checkpoint_dir = util.io.get_dir(FLAGS.checkpoint_path)
    logdir = util.io.join_path(FLAGS.checkpoint_path, 'test', FLAGS.dataset_name + '_' +FLAGS.dataset_split_name)
    
    saver = tf.train.Saver()
    if util.io.is_dir(FLAGS.checkpoint_path):
        checkpoint = util.tf.get_latest_ckpt(FLAGS.checkpoint_path)
    else:
        checkpoint = FLAGS.checkpoint_path
        
    tf.logging.info('testing', checkpoint)

    with tf.Session(config = sess_config) as sess:
        saver.restore(sess, checkpoint)
        checkpoint_name = util.io.get_filename(str(checkpoint));
        dump_path = util.io.join_path(logdir, checkpoint_name, 
                                      'seg_link_conf_th_%f_%f'%(config.seg_conf_threshold, config.link_conf_threshold))
        
        txt_path = util.io.join_path(dump_path,'txt')
        zip_path = util.io.join_path(dump_path, '%s_seg_link_conf_th_%f_%f.zip'%(checkpoint_name, config.seg_conf_threshold, config.link_conf_threshold))
        
        # write detection result as txt files
        def write_result_as_txt(image_name, bboxes, path):
          filename = util.io.join_path(path, 'res_%s.txt'%(image_name))
          lines = []
          for b_idx, bbox in enumerate(bboxes):
                values = [int(v) for v in bbox]
                line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
                lines.append(line)
          util.io.write_lines(filename, lines)
          print 'result has been written to:', filename
          
        for iter, image_name in enumerate(image_names):
            image_data = util.img.imread(util.io.join_path(FLAGS.dataset_dir, image_name), rgb = True)
            image_name = image_name.split('.')[0]
            image_bboxes = sess.run([bboxes_pred], feed_dict = {image:image_data, image_shape:image_data.shape})
            print '%d/%d: %s'%(iter + 1, len(image_names), image_name)
            write_result_as_txt(image_name, image_bboxes[0], txt_path)
                
        # create zip file for icdar2015
        cmd = 'cd %s;zip -j %s %s/*'%(dump_path, zip_path, txt_path);
        print cmd
        print util.cmd.cmd(cmd);
        print "zip file created: ", util.io.join_path(dump_path, zip_path)


def main(_):
    config_initialization()
    eval()
    
    
if __name__ == '__main__':
    tf.app.run()
