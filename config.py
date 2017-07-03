from __future__ import print_function
from pprint import pprint
import numpy as np
from tensorflow.contrib.slim.python.slim.data import parallel_reader
import tensorflow as tf
slim = tf.contrib.slim
import util


global feat_shapes
global image_shape


global default_anchors
global defalt_anchor_map
global default_anchor_center_set
global num_anchors
global num_links


global batch_size
global batch_size_per_gpu
global gpus
global num_clones
global clone_scopes


global train_with_ignored
global seg_loc_loss_weight
global conf_cls_loss_weight

global seg_conf_threshold
global link_conf_threshold

anchor_offset = 0.5    
anchor_scale_gamma = 1.5
feat_layers = ['conv4_3','fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
# feat_norms = [20] + [-1] * len(feat_layers)
max_height_ratio = 1.5
# prior_scaling = [0.1, 0.2, 0.1, 0.2, 20.0]
prior_scaling = [0.2, 0.5, 0.2, 0.5, 20.0]
# prior_scaling = [1.0] * 5

max_neg_pos_ratio = 3

data_format = 'NHWC'
def _set_image_shape(shape):
    global image_shape
    image_shape = shape

def _set_feat_shapes(shapes):
    global feat_shapes
    feat_shapes = shapes

def _set_batch_size(bz):
    global batch_size
    batch_size = bz

def _set_det_th(seg_conf_th, link_conf_th):
    global seg_conf_threshold
    global link_conf_threshold
    
    seg_conf_threshold = seg_conf_th
    link_conf_threshold = link_conf_th
    
def  _set_loss_weight(seg_loc_loss_w, link_cls_loss_w):
    global seg_loc_loss_weight
    global link_cls_loss_weight
    seg_loc_loss_weight = seg_loc_loss_w
    link_cls_loss_weight = link_cls_loss_w
    
def  _set_train_with_ignored(train_with_ignored_):
    global train_with_ignored    
    train_with_ignored = train_with_ignored_

def _build_anchor_map():
    global default_anchor_map
    global default_anchor_center_set
    import collections
    default_anchor_map = collections.defaultdict(list)
    for anchor_idx, anchor in enumerate(default_anchors):
        default_anchor_map[(int(anchor[1]), int(anchor[0]))].append(anchor_idx)
    default_anchor_center_set = set(default_anchor_map.keys())
    
def init_config(image_shape, batch_size = 1, 
                weight_decay = 0.0005, 
                num_gpus = 1, 
                train_with_ignored = False,
                seg_loc_loss_weight = 1.0,
                link_cls_loss_weight = 1.0,
                seg_conf_threshold = 0.5,
                link_conf_threshold = 0.5):

    _set_det_th(seg_conf_threshold, link_conf_threshold)
    _set_loss_weight(seg_loc_loss_weight, link_cls_loss_weight)
    _set_train_with_ignored(train_with_ignored)

    h, w = image_shape
    from nets import anchor_layer
    from nets import seglink_symbol
    fake_image = tf.ones((1, h, w, 3))
    fake_net = seglink_symbol.SegLinkNet(inputs = fake_image, weight_decay = weight_decay)
    feat_shapes = fake_net.get_shapes();
    
    # the placement of the following lines are extremely important
    _set_image_shape(image_shape)
    _set_feat_shapes(feat_shapes)

    anchors, _ = anchor_layer.generate_anchors()
    global default_anchors
    default_anchors = anchors
    
    global num_anchors
    num_anchors = len(anchors)
    
    _build_anchor_map()
    
    global num_links
    num_links = num_anchors * 8 + (num_anchors - np.prod(feat_shapes[feat_layers[0]])) * 4
    
    #init batch size
    global gpus
    gpus = util.tf.get_available_gpus(num_gpus)
    
    global num_clones
    num_clones = len(gpus)
    
    global clone_scopes
    clone_scopes = ['clone_%d'%(idx) for idx in xrange(num_clones)]
    
    _set_batch_size(batch_size)
    
    global batch_size_per_gpu
    batch_size_per_gpu = batch_size / num_clones
    if batch_size_per_gpu < 1:
        raise ValueError('Invalid batch_size [=%d], resulting in 0 images per gpu.'%(batch_size))
    
    
def print_config(flags, dataset, save_dir = None, print_to_file = True):
    def do_print(stream=None):
        print('\n# =========================================================================== #', file=stream)
        print('# Training flags:', file=stream)
        print('# =========================================================================== #', file=stream)
        pprint(flags.__flags, stream=stream)

        print('\n# =========================================================================== #', file=stream)
        print('# seglink net parameters:', file=stream)
        print('# =========================================================================== #', file=stream)
        vars = globals()
        for key in vars:
            var = vars[key]
            if util.dtype.is_number(var) or util.dtype.is_str(var) or util.dtype.is_list(var) or util.dtype.is_tuple(var):
                pprint('%s=%s'%(key, str(var)), stream = stream)
            
        print('\n# =========================================================================== #', file=stream)
        print('# Training | Evaluation dataset files:', file=stream)
        print('# =========================================================================== #', file=stream)
        data_files = parallel_reader.get_data_files(dataset.data_sources)
        pprint(sorted(data_files), stream=stream)
        print('', file=stream)
    do_print(None)
    
    if print_to_file:
        # Save to a text file as well.
        if save_dir is None:
            save_dir = flags.train_dir
            
        util.io.mkdir(save_dir)
        path = util.io.join_path(save_dir, 'training_config.txt')
        with open(path, "a") as out:
            do_print(out)
    
