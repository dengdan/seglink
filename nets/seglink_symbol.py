import tensorflow as tf
import tensorflow.contrib.slim as slim
import net_factory

class SegLinkNet(object):
    def __init__(self, inputs, feat_layers, basenet_type = 'vgg'):
        self.inputs = inputs;
        self.feat_layers = feat_layers
        self.basenet_type = basenet_type;
        self._build_network();
        self.shapes = self.get_shapes();
    def get_shapes(self):
        shapes = {}
            
        for layer in self.end_points:
            shapes[layer] = tensor_shape(self.end_points[layer])[1:-1]
            
        return shapes
    def get_shape(self, name):
        return self.shapes[name] 
    def _build_network(self):
        with tf.variable_scope(self.basenet_type):
            basenet, end_points = net_factory.get_basenet(self.basenet_type, self.inputs);
            
        with tf.variable_scope('extra_layers'):
            self.net, self.end_points = self._add_extra_layers(basenet, end_points);
        
        with tf.variable_scope('seg_link_layers'):
            self._add_seg_link_layers();
        
    def _add_extra_layers(self, inputs, end_points):
        # Additional SSD blocks.
        # conv6/7/8/9/10: 1x1 and 3x3 convolutions stride 2 (except lasts).
        net = slim.conv2d(inputs, 256, [1, 1], scope='conv6_1')
        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv6_2', padding='SAME')
        end_points['conv6_2'] = net
        
        net = slim.conv2d(net, 128, [1, 1], scope='conv7_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv7_2', padding='SAME')
        end_points['conv7_2'] = net

        net = slim.conv2d(net, 128, [1, 1], scope='conv8_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv8_2', padding='SAME')
        end_points['conv8_2'] = net

        net = slim.conv2d(net, 128, [1, 1], scope='conv9_1')
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv9_2', padding='SAME')
        end_points['conv9_2'] = net
        
        
        net = slim.conv2d(net, 128, [1, 1], scope='conv10_1')
        net = slim.conv2d(net, 256, [2, 2], scope='conv10_2', padding='VALID')
        end_points['conv10_2'] = net
        return net, end_points;    
    
    def _build_seg_link_layer(self, layer_name):
        net = self.end_points[layer_name]
        batch_size, h, w = tensor_shape(net)[:-1]
        
        # segment scores
        num_cls_pred = 2
        seg_scores = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn = None, scope='seg_scores')
        
        # segment offsets
        num_offset_pred = 5
        seg_offsets = slim.conv2d(net, num_offset_pred, [3, 3], activation_fn = None, scope = 'seg_offsets')
        
        # within-layer link scores
        num_within_layer_link_scores_pred = 16
        within_layer_link_scores = slim.conv2d(net, num_within_layer_link_scores_pred, [3, 3], activation_fn = None, scope = 'within_layer_link_scores')
        within_layer_link_scores = tf.reshape(within_layer_link_scores, tensor_shape(within_layer_link_scores)[:-1] + [8, 2])

        # cross-layer link scores
        num_cross_layer_link_scores_pred = 8
        cross_layer_link_scores = None;
        if layer_name != 'conv4_3':
            cross_layer_link_scores = slim.conv2d(net, num_cross_layer_link_scores_pred, [3, 3], activation_fn = None, scope = 'cross_layer_link_scores')
            cross_layer_link_scores = tf.reshape(cross_layer_link_scores, tensor_shape(cross_layer_link_scores)[:-1] + [4, 2])

        return seg_scores, seg_offsets, within_layer_link_scores, cross_layer_link_scores
    
    
    def _add_seg_link_layers(self):
        all_seg_scores = []
        all_seg_offsets = []
        all_within_layer_link_scores = []
        all_cross_layer_link_scores  = []
        for layer_name in self.feat_layers:
            with tf.variable_scope(layer_name):
                seg_scores, seg_offsets, within_layer_link_scores, cross_layer_link_scores = self._build_seg_link_layer(layer_name)
            all_seg_scores.append(seg_scores)
            all_seg_offsets.append(seg_offsets)
            all_within_layer_link_scores.append(within_layer_link_scores)
            all_cross_layer_link_scores.append(cross_layer_link_scores)
        self.seg_scores = reshape_and_concat(all_seg_scores) # (batch_size, N, 2)
        self.seg_offsets = reshape_and_concat(all_seg_offsets) # (batch_size, N, 5)
        self.cross_layer_link_scores = reshape_and_concat(all_cross_layer_link_scores)  # (batch_size, 8N, 2)
        self.within_layer_link_scores = reshape_and_concat(all_within_layer_link_scores)  # (batch_size, 4(N - N_conv4_3), 2)
        self.link_scores = tf.concat([self.within_layer_link_scores, self.cross_layer_link_scores], axis = 1)
        
        
def reshape_and_concat(tensors):
    def reshape(t):
        shape = tensor_shape(t)
        if len(shape) == 4:
            shape = (shape[0], -1, shape[-1])
            t = tf.reshape(t, shape)
        elif len(shape) == 5:
            shape = (shape[0], -1, shape[-2], shape[-1])
            t = tf.reshape(t, shape)
            t = tf.reshape(t, [shape[0], -1, shape[-1]])
        else:
            raise ValueError("invalid tensor shape: %s, shape = %s"%(t.name, shape)) 
        return t;                   
    reshaped_tensors = [reshape(t) for t in tensors if t is not None]
    return tf.concat(reshaped_tensors, axis = 1)
    
def tensor_shape(t):
    t.get_shape().assert_is_fully_defined()
    return t.get_shape().as_list()
    
