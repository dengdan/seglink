import numpy as np
import config 
def generate_anchors():
    """
    
    """
    all_anchors = []
    layer_anchors = {}
    h_I, w_I = config.image_shape;
    for layer_name in config.feat_layers:
        feat_shape = config.feat_shapes[layer_name];
        h_l, w_l = feat_shape
        anchors = _generate_anchors_one_layer(h_I, w_I, h_l, w_l)
        all_anchors.append(anchors)
        layer_anchors[layer_name] = anchors
    all_anchors = _reshape_and_concat(all_anchors)
    return all_anchors, layer_anchors
    
    
def _reshape_and_concat(tensors):
    tensors = [np.reshape(t, (-1, t.shape[-1])) for t in tensors]
    return np.vstack(tensors)
        
def _generate_anchors_one_layer(h_I, w_I, h_l, w_l):
    """
    generate anchors on on layer
    return a ndarray with shape (h_l, w_l, 4), and the last dimmension in the order:[cx, cy, w, h]
    """
    y, x = np.mgrid[0: h_l, 0:w_l]
    cy = (y + config.anchor_offset) / h_l * h_I
    cx = (x + config.anchor_offset) / w_l * w_I
    
    anchor_scale = _get_scale(w_I, w_l)
    anchor_w = np.ones_like(cx) * anchor_scale
    anchor_h = np.ones_like(cx) * anchor_scale # cx.shape == cy.shape
    
    anchors = np.asarray([cx, cy, anchor_w, anchor_h])
    anchors = np.transpose(anchors, (1, 2, 0))
    
    return anchors
    
    
def _get_scale(w_I, w_l):
    return config.anchor_scale_gamma * 1.0 * w_I / w_l
    
    

def _test_generate_anchors_one_layer():
    """
    test _generate_anchors_one_layer method by visualizing it in an image.
    """
    import util
    image_shape = (512, 512)
    h_I, w_I = image_shape
    stride = 256
    feat_shape = (h_I/stride, w_I / stride)
    h_l, w_l = feat_shape
    anchors = _generate_anchors_one_layer(h_I, w_I, h_l, w_l, gamma = 1.5)
    assert(anchors.shape == (h_l, w_l, 4))
    mask = util.img.black(image_shape)
    for x in xrange(w_l):
        for y in xrange(h_l):
            cx, cy, w, h = anchors[y, x, :]
            xmin = (cx - w / 2)
            ymin = (cy - h / 2)
            
            xmax = (cx + w / 2)
            ymax = (cy + h / 2)
            
            cxy = (int(cx), int(cy))
            util.img.circle(mask, cxy, 3, color = 255)
            util.img.rectangle(mask, (xmin, ymin), (xmax, ymax), color = 255)
    
    util.sit(mask)
            
    
if __name__ == "__main__":
    _test_generate_anchors_one_layer();
