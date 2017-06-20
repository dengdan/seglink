import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import util

global feat_shapes
global image_shape
global default_anchors
global num_anchors
global num_links
global batch_size
global batch_size_per_gpu
global gpus
global num_clones
global clone_scopes

anchor_offset = 0.5    
anchor_scale_gamma = 1.5
feat_layers = ['conv4_3','fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2', 'conv10_2']

max_height_ratio = 1.5 * 2


seg_confidence_threshold = 0.5
link_confidence_threshold = 0.5

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
    
def init_config(image_shape, batch_size = 1, weight_decay = None):
    from nets import anchor_layer
    from nets import seglink_symbol

    h, w = image_shape
    
    collections = {}
    
    keys = [v for v in tf.GraphKeys.__dict__ if not v.startswith('__')]
    for key in keys:
        try:
            collections[key] = tf.get_collection(key)
        except:
            print key
        
    fake_image = tf.ones((1, h, w, 3))
    
    fake_net = seglink_symbol.SegLinkNet(inputs = fake_image, weight_decay = weight_decay)
    # TODO: how to delete these created temp variables 
    feat_shapes = fake_net.get_shapes();
    
            
    print 'image_shape:', image_shape
    print 'feat_shapes:',feat_shapes
    
    # the placement of this following lines are extremely important
    _set_image_shape(image_shape)
    _set_feat_shapes(feat_shapes)

    anchors, _ = anchor_layer.generate_anchors()
    global default_anchors
    default_anchors = anchors
    
    global num_anchors
    num_anchors = len(anchors)
    
    global num_links
    num_links = num_anchors * 8 + (num_anchors - np.prod(feat_shapes[feat_layers[0]])) * 4
    
    #init batch size
    global gpus
    gpus = util.tf.get_available_gpus()
    
    global num_clones
    num_clones = len(gpus)
    
    global clone_scopes
    clone_scopes = ['clone_%d'%(idx) for idx in xrange(num_clones)]
    
    _set_batch_size(batch_size)
    
    global batch_size_per_gpu
    batch_size_per_gpu = batch_size / num_clones
    if batch_size_per_gpu < 1:
        raise ValueError('Invalid batch_size [=%d], resulting in 0 images per gpu.'%(batch_size))
    
