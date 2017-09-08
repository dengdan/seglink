import tensorflow as tf
import numpy as np
import caffe
from caffe.proto import caffe_pb2
import util
from nets import seglink_symbol

caffemodel_path=util.io.get_absolute_path('~/models/ssd-pretrain/VGG_coco_SSD_512x512_iter_360000.caffemodel')
class CaffeScope():
    def __init__(self):
        print('Loading Caffe file:', caffemodel_path)
        caffemodel_params = caffe_pb2.NetParameter()
        caffemodel_str = open(caffemodel_path, 'rb').read()
        caffemodel_params.ParseFromString(caffemodel_str)
        caffe_layers = caffemodel_params.layer
        self.layers = []
        self.counter = 0
        self.bgr_to_rgb = False
        for layer in caffe_layers:
            if layer.type == 'Convolution':
                self.layers.append(layer)
        
    def conv_weights_init(self):
        def _initializer(shape, dtype, partition_info=None):
            layer = self.layers[self.counter]
            w = np.array(layer.blobs[0].data)
            # Weights: reshape and transpose dimensions.
            w = np.reshape(w, layer.blobs[0].shape.dim)
            # w = np.transpose(w, (1, 0, 2, 3))
            w = np.transpose(w, (2, 3, 1, 0))
            if self.bgr_to_rgb and w.shape[2] == 3:
                print('Convert BGR to RGB in convolution layer:', layer.name)
                w[:, :, (0, 1, 2)] = w[:, :, (2, 1, 0)]
                self.bgr_to_rgb = False
            np.testing.assert_equal(w.shape, shape)
            print('Load weights from convolution layer:', layer.name, w.shape, shape)
            return tf.cast(w, dtype)

        return _initializer
    
    def conv_biases_init(self):
        def _initializer(shape, dtype, partition_info=None):
            layer = self.layers[self.counter]
            self.counter = self.counter + 1
            b = np.array(layer.blobs[1].data)

            print('Load biases from convolution layer:', layer.name, b.shape)
            return tf.cast(b, dtype)
        return _initializer


caffe_scope = CaffeScope()

# def get_seglink_model():
fake_image = tf.placeholder(dtype = tf.float32, shape = [1, 512, 1024, 3])
seglink_net = seglink_symbol.SegLinkNet(inputs = fake_image, weight_decay = 0.01, 
                                        weights_initializer = caffe_scope.conv_weights_init(), biases_initializer = caffe_scope.conv_biases_init())
init_op = tf.global_variables_initializer()
with tf.Session() as session:
    # Run the init operation.
    session.run(init_op)

    # Save model in checkpoint.
    saver = tf.train.Saver(write_version=2)
    parent_dir = util.io.get_dir(caffemodel_path)
    filename = util.io.get_filename(caffemodel_path)
    parent_dir = util.io.mkdir(util.io.join_path(parent_dir, 'seglink'))
    filename = filename.replace('.caffemodel', '.ckpt')
    ckpt_path = util.io.join_path(parent_dir, filename)
    saver.save(session, ckpt_path, write_meta_graph=False)
    
    vars = tf.global_variables()            
    layers_to_convert = ['conv1_1', 'conv1_2', 
                     'conv2_1', 'conv2_2', 
                     'conv3_1', 'conv3_2', 'conv3_3', 
                     'conv4_1', 'conv4_2', 'conv4_3', 
                     'conv5_1', 'conv5_2', 'conv5_3', 
                     'fc6', 'fc7', 
                     'conv6_1', 'conv6_2', 
                     'conv7_1', 'conv7_2', 
                     'conv8_1', 'conv8_2', 
                     'conv9_1', 'conv9_2', 
#                      'conv10_1', 'conv10_2'
                     ]

    def check_var(name):
        tf_weights = None
        tf_biases = None
        
        for var in vars:
            if util.str.contains(str(var.name), name) and util.str.contains(str(var.name), 'weight') and not util.str.contains(str(var.name), 'seglink'):
                tf_weights = var

            if util.str.contains(str(var.name), name) and util.str.contains(str(var.name), 'bias') and not util.str.contains(str(var.name), 'seglink'):
                tf_biases = var
        
        caffe_weights = None
        caffe_biases = None
        for layer in caffe_scope.layers:
            if name == layer.name:
                caffe_weights = layer.blobs[0].data
                caffe_biases =  layer.blobs[1].data
        
        np.testing.assert_almost_equal(actual = np.mean(caffe_weights), desired = np.mean(tf_weights.eval(session)))
        np.testing.assert_almost_equal(actual = np.mean(caffe_biases), desired = np.mean(tf_biases.eval(session)))
    
    # check all vgg and extra layer weights/biases have been converted in a right way.
    for name in layers_to_convert:
        check_var(name)
    
    # just have peek into the values of seglink layers. The weights should not be initialized to 0. Just have a look.
    for var in vars:
        if util.str.contains(str(var.name), 'seglink'):
            print var.name, np.mean(var.eval(session)), np.std(var.eval(session))
    