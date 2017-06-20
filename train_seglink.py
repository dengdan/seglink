#test code to make sure the ground truth calculation and data batch works well.

import numpy as np
import tensorflow as tf # test

from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from tf_extended import seglink
import util
import cv2
from nets import seglink_symbol, anchor_layer


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


tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer')
tf.app.flags.DEFINE_float('weight_decay', 0.0005, 'The weight decay on the model weights.')

FLAGS = tf.app.flags.FLAGS

def config_initialization():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    
    
    # image shape and feature layers shape inference
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)
    
    config.init_config(image_shape, batch_size = FLAGS.batch_size, weight_decay = FLAGS.weight_decay)

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

def sum_gradients(clone_grads):                        
    averaged_grads = []
    for grad_and_vars in zip(*clone_grads):
        grads = []
        v = grad_and_vars[0][1]
        for g, _ in grad_and_vars:
            grads.append(g)
        grad = tf.add_n(grads, v.name + '_summed_gradients')
        averaged_grads.append((grad, v))
    return averaged_grads


def create_clones(batch_queue):        
    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()
        learning_rate = tf.constant(FLAGS.learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum, name='Momentum')

    # place clones
    total_loss = 0; # for summary only
    gradients = []
    for clone_idx, gpu in enumerate(config.gpus):
        do_summary = clone_idx == 0 # only summary on the first clone
        with tf.variable_scope(tf.get_variable_scope(), reuse = True):# the variables has been created in config.init_config
            with tf.name_scope(config.clone_scopes[clone_idx]) as clone_scope:
                with tf.device(gpu) as clone_device:
                    b_image, b_seg_label, b_seg_loc, b_link_gt = batch_queue.dequeue()
                    net = seglink_symbol.SegLinkNet(inputs = b_image, data_format = config.data_format)
                    
                    # build seglink loss
                    net.build_loss(seg_label = b_seg_label, seg_loc = b_seg_loc, link_label = b_link_gt,
                                  seg_loc_loss_weight = 1.0, link_conf_loss_weight = 1.0, do_summary = do_summary)
                    
                    # gather seglink losses
                    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone_scope)
                    assert len(losses) ==  3  # 3 is the number of seglink losses: seg_cls, seg_loc, link_cls
                    total_clone_loss = tf.add_n(losses) / config.num_clones
                    total_loss = total_loss + total_clone_loss

                    # gather regularization loss and add to clone_0 only
                    if clone_idx == 0:
                        import pdb
                        pdb.set_trace()
                        regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        total_clone_loss = total_clone_loss + regularization_loss
                    
                    # compute clone gradients
                    clone_gradients = optimizer.compute_gradients(total_clone_loss)# all variables will be updated.
                    gradients.append(clone_gradient)
    
    # add all gradients together
    # note that the gradients do not need to be averaged, because the average operation has been done on loss.
    averaged_gradients = sum_gradients(gradients)
    
    train_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)
    return train_op

    
    
def train(train_op):
    merged = tf.summary.merge_all()
    summary_op = tf.get_collection(tf.GraphKeys.SUMMARIES)
    import pdb
    pdb.set_trace()    
    
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;
    
    saver = tf.train.Saver(max_to_keep=500, write_version=2, pad_step_number=False)
    slim.learning.train(
            train_op,
            logdir=FLAGS.train_dir,
            init_fn=util.tf.get_init_fn(FLAGS),
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs= FLAGS.save_summaries_secs,
            saver=saver,
            save_interval_secs=FLAGS.save_interval_secs,
            session_config=sess_config
    )


def main(_):
    config_initialization()
    batch_queue = create_dataset_batch_queue()
    train_op = create_clones(batch_queue)
    train(train_op)
    
    
if __name__ == '__main__':
    tf.app.run()
