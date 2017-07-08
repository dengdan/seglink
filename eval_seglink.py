#encoding = utf-8

import numpy as np
import math
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.training.python.training import evaluation
from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from tf_extended import seglink, metrics as tfe_metrics, bboxes as tfe_bboxes
import util
import cv2
from nets import seglink_symbol, anchor_layer


slim = tf.contrib.slim
import config
# =========================================================================== #
# model threshold parameters
# =========================================================================== #
tf.app.flags.DEFINE_string('train_with_ignored', False, 
       'whether to use ignored bbox (in ic15) in training.')
tf.app.flags.DEFINE_boolean('do_grid_search', False, 
       'whether to do grid search to find a best combinations of \
       seg_conf_threshold and link_conf_threshold.')
tf.app.flags.DEFINE_float('seg_loc_loss_weight', 1.0, 
      'the loss weight of segment localization')
tf.app.flags.DEFINE_float('link_cls_loss_weight', 1.0, 
      'the loss weight of linkage classification loss')

tf.app.flags.DEFINE_float('seg_conf_threshold', 0.9, 
      'the threshold on the confidence of segment')
tf.app.flags.DEFINE_float('link_conf_threshold', 0.7, 
      'the threshold on the confidence of linkage')


# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of checkpoint to be evaluated. \
   If it is a directory containing many checkpoints, \
   the lastest will be evaluated.')
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.1, 
  'the gpu memory fraction to be used. \
   If less than 0, allow_growth = True is used.')
tf.app.flags.DEFINE_bool('using_moving_average', False, 
   'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 
    'The decay rate of ExponentionalMovingAverage')

# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
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
    
    config.init_config(image_shape, 
                       batch_size = 1, 
                       seg_conf_threshold = FLAGS.seg_conf_threshold,
                       link_conf_threshold = FLAGS.link_conf_threshold, 
                       train_with_ignored = FLAGS.train_with_ignored,
                       seg_loc_loss_weight = FLAGS.seg_loc_loss_weight, 
                       link_cls_loss_weight = FLAGS.link_cls_loss_weight, 
                       )
        
    
    util.proc.set_proc_name('eval_' + FLAGS.model_name + '_' + FLAGS.dataset_name )
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    config.print_config(FLAGS, dataset, print_to_file = False)
    
    return dataset

def read_dataset(dataset):
    with tf.name_scope(FLAGS.dataset_name +'_'  + FLAGS.dataset_split_name + '_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            shuffle=False)
        
    [image, shape, filename, gignored, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get([
                                                     'image', 'shape', 'filename',
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
                                                       is_training = False)
    image = tf.identity(image, 'processed_image')
    
    # calculate ground truth
    seg_label, seg_loc, link_gt = seglink.tf_get_all_seglink_gt(gxs, gys, gignored)
        
    return image, seg_label, seg_loc, link_gt, filename, shape, gignored, gxs, gys

def eval(dataset):
    dict_metrics = {} 
    checkpoint_dir = util.io.get_dir(FLAGS.checkpoint_path)
    logdir = util.io.join_path(checkpoint_dir, 
                               'eval',  
                               "%s_%s"%(FLAGS.dataset_name, FLAGS.dataset_split_name))
    
    global_step = slim.get_or_create_global_step()
    with tf.name_scope('evaluation_%dx%d'%(FLAGS.eval_image_height, FLAGS.eval_image_width)):
        with tf.variable_scope(tf.get_variable_scope(), reuse = True):# the variables has been created in config.init_config
            # get input tensor
            image, seg_label, seg_loc, link_gt, filename, shape, gignored, gxs, gys = read_dataset(dataset)
            # expand dim if needed
            b_image =  tf.expand_dims(image, axis = 0);
            b_seg_label = tf.expand_dims(seg_label, axis = 0)
            b_seg_loc = tf.expand_dims(seg_loc, axis = 0)
            b_link_gt = tf.expand_dims(link_gt, axis = 0)
            b_shape = tf.expand_dims(shape, axis = 0)
            
            # build seglink loss
            net = seglink_symbol.SegLinkNet(inputs = b_image, data_format = config.data_format)
            net.build_loss(seg_labels = b_seg_label, 
                           seg_offsets = b_seg_loc, 
                           link_labels = b_link_gt,
                           do_summary = False) # the summary will be added in the following lines
            
            # gather seglink losses
            losses = tf.get_collection(tf.GraphKeys.LOSSES)
            assert len(losses) ==  3  # 3 is the number of seglink losses: seg_cls, seg_loc, link_cls
            for loss in tf.get_collection(tf.GraphKeys.LOSSES):
                dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)
            
            seglink_loss = tf.add_n(losses)
            dict_metrics['seglink_loss'] = slim.metrics.streaming_mean(seglink_loss)
            
            # Add metrics to summaries.
            for name, metric in dict_metrics.items():
                tf.summary.scalar(name, metric[0])
            
            # shape = (height, width, channels) when format = NHWC TODO
            gxs = gxs * tf.cast(shape[1], gxs.dtype)
            gys = gys * tf.cast(shape[0], gys.dtype)
            if FLAGS.do_grid_search:
                # grid search            
                seg_ths = np.arange(0.5, 0.91, 0.1)
                link_ths = seg_ths
            else:
                seg_ths = [FLAGS.seg_conf_threshold]
                link_ths = [FLAGS.link_conf_threshold]
            
            eval_result_path = util.io.join_path(logdir, 'eval_on_%s_%s.log'%(FLAGS.dataset_name, FLAGS.dataset_split_name))
            for seg_th in seg_ths:
                for link_th in link_ths:
                    config._set_det_th(seg_th, link_th)
                    
                    eval_result_msg = 'seg_conf_threshold=%f, link_conf_threshold = %f, '\
                                            %(config.seg_conf_threshold, config.link_conf_threshold)
                    eval_result_msg += 'iter = %r, recall = %r, precision = %f, fmean = %r'
                    
                    with tf.name_scope('seglink_conf_th_%f_%f'\
                                       %(config.seg_conf_threshold, config.link_conf_threshold)):
                        # decode seglink to bbox output, with absolute length, instead of being within [0,1]
                        bboxes_pred = seglink.tf_seglink_to_bbox(net.seg_scores, net.link_scores, net.seg_offsets,
                                                                  b_shape, seg_conf_threshold = seg_th, link_conf_threshold = link_th)
#                         bboxes_pred = tf.Print(bboxes_pred, [tf.shape(bboxes_pred)], '%f_%f, shape of bboxes = '%(seg_th, link_th))
                        # calculate true positive and false positive
                        # the xs and ys from tfrecord is 0~1, resize them to absolute length before matching.
                        num_gt_bboxes, tp, fp = tfe_bboxes.bboxes_matching(bboxes_pred, gxs, gys, gignored)
                        tp_fp_metric = tfe_metrics.streaming_tp_fp_arrays(num_gt_bboxes, tp, fp)
                        dict_metrics['tp_fp_%f_%f'%(config.seg_conf_threshold, config.link_conf_threshold)] = (tp_fp_metric[0], tp_fp_metric[1])
                         
                        # precision and recall
                        precision, recall = tfe_metrics.precision_recall(*tp_fp_metric[0])
                         
                        fmean = tfe_metrics.fmean(precision, recall)
                        fmean = util.tf.Print(fmean, data = [global_step, recall, precision, fmean], 
                                                msg = eval_result_msg, 
                                                file = eval_result_path, mode = 'a')
                        fmean = tf.Print(fmean, [recall, precision, fmean], '%f_%f, Recall, Precision, Fmean = '%(seg_th, link_th))
                        tf.summary.scalar('Precision', precision)
                        tf.summary.scalar('Recall', recall)
                        tf.summary.scalar('F-mean', fmean)
            
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)

    
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;
    
    # Variables to restore: moving avg. or normal weights.
    if FLAGS.using_moving_average:
        variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
        variables_to_restore[global_step.op.name] = global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()

    if util.io.is_dir(FLAGS.checkpoint_path):
        slim.evaluation.evaluation_loop(
            master = '',
            eval_op=list(names_to_updates.values()),
            num_evals=dataset.num_samples,
            variables_to_restore=variables_to_restore,
            checkpoint_dir = checkpoint_dir,
            logdir = logdir,
            session_config=sess_config)
    else:
        slim.evaluation.evaluate_once(
            master = '',
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore,
            num_evals=2,#dataset.num_samples,
            checkpoint_path = FLAGS.checkpoint_path,
            logdir = logdir,
            session_config=sess_config)

                
        
def main(_):
    eval(config_initialization())
    
    
if __name__ == '__main__':
    tf.app.run()
