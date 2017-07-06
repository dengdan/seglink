import cv2
import numpy as np
import tensorflow as tf

import config
import util

############################################################################################################
#                       seg_gt calculation                                                                 #
############################################################################################################

def anchor_rect_height_ratio(anchor, rect):
    """calculate the height ratio between anchor and rect
    """
    rect_height = min(rect[2], rect[3])
    anchor_height = anchor[2] * 1.0
    ratio = anchor_height / rect_height
    return max(ratio, 1.0 / ratio)
    
def is_anchor_center_in_rect(anchor, xs, ys, bbox_idx):
    """tell if the center of the anchor is in the rect represented using xs and ys and bbox_idx 
    """
    bbox_points = zip(xs[bbox_idx, :], ys[bbox_idx, :])
    cnt = util.img.points_to_contour(bbox_points);
    acx, acy, aw, ah = anchor
    return util.img.is_in_contour((acx, acy), cnt)
    
def min_area_rect(xs, ys):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta]. 
    """
    xs = np.asarray(xs, dtype = np.float32)
    ys = np.asarray(ys, dtype = np.float32)
        
    num_rects = xs.shape[0]
    box = np.empty((num_rects, 5))#cx, cy, w, h, theta
    for idx in xrange(num_rects):
        points = zip(xs[idx, :], ys[idx, :])
        cnt = util.img.points_to_contour(points)
        rect = cv2.minAreaRect(cnt)
        cx, cy = rect[0]
        w, h = rect[1]
        theta = rect[2]
        box[idx, :] = [cx, cy, w, h, theta]
    
    box = np.asarray(box, dtype = xs.dtype)
    return box

def tf_min_area_rect(xs, ys):
    return tf.py_func(min_area_rect, [xs, ys], xs.dtype)

def transform_cv_rect(rects):
    """Transform the rects from opencv method minAreaRect to our rects. 
    Step 1 of Figure 5 in seglink paper

    In cv2.minAreaRect, the w, h and theta values in the returned rect are not convenient to use (at least for me), so 
            the Oriented (or rotated) Rectangle object in seglink algorithm is defined different from cv2.
    
    Rect definition in Seglink:
        1. The angle value between a side and x-axis is:
            positive: if it rotates clockwisely, with y-axis increasing downwards.
            negative: if it rotates counter-clockwisely.
            This is opposite to cv2, and it is only a personal preference. 
        
        2. The width is the length of side taking a smaller absolute angle with the x-axis. 
        3. The theta value of a rect is the signed angle value between width-side and x-axis
        4. To rotate a rect to horizontal direction, just rotate its width-side horizontally,
             i.e., rotate it by a angle of theta using cv2 method. 
             (see the method rotate_oriented_bbox_to_horizontal for rotation detail)
            
    
    Args:
        rects: ndarray with shape = (5, ) or (N, 5).
    Return:
        transformed rects.
    """
    only_one = False
    if len(np.shape(rects)) == 1:
        rects = np.expand_dims(rects, axis = 0)
        only_one = True
    assert np.shape(rects)[1] == 5, 'The shape of rects must be (N, 5), but meet %s'%(str(np.shape(rects)))
    
    rects = np.asarray(rects, dtype = np.float32).copy()
    num_rects = np.shape(rects)[0]
    for idx in xrange(num_rects):
        cx, cy, w, h, theta = rects[idx, ...];
        #assert theta < 0 and theta >= -90, "invalid theta: %f"%(theta) 
        if abs(theta) > 45 or (abs(theta) == 45 and w < h):
            w, h = [h, w]
            theta = 90 + theta
        rects[idx, ...] = [cx, cy, w, h, theta]
    if only_one:
        return rects[0, ...]
    return rects                
    

def rotate_oriented_bbox_to_horizontal(center, bbox):
    """
    Step 2 of Figure 5 in seglink paper
    
    Rotate bbox horizontally along a `center` point
    Args:
        center: the center of rotation
        bbox: [cx, cy, w, h, theta]
    """
    assert np.shape(center) == (2, ), "center must be a vector of length 2"
    assert np.shape(bbox) == (5, ) or np.shape(bbox) == (4, ), "bbox must be a vector of length 4 or 5"
    bbox = np.asarray(bbox.copy(), dtype = np.float32)
    
    cx, cy, w, h, theta = bbox;
    M = cv2.getRotationMatrix2D(center, theta, scale = 1) # 2x3
    
    cx, cy = np.dot(M, np.transpose([cx, cy, 1]))
    
    bbox[0:2] = [cx, cy] 
    return bbox

def crop_horizontal_bbox_using_anchor(bbox, anchor):
    """Step 3 in Figure 5 in seglink paper
    The crop operation is operated only on the x direction.
    Args:
        bbox: a horizontal bbox with shape = (5, ) or (4, ). 
    """
    assert np.shape(anchor) == (4, ), "anchor must be a vector of length 4"
    assert np.shape(bbox) == (5, ) or np.shape(bbox) == (4, ), "bbox must be a vector of length 4 or 5"
    
    # xmin and xmax of the anchor    
    acx, acy, aw, ah = anchor
    axmin = acx - aw / 2.0;
    axmax = acx + aw / 2.0;
    
    # xmin and xmax of the bbox
    cx, cy, w, h = bbox[0:4]
    xmin = cx - w / 2.0
    xmax = cx + w / 2.0
    
    # clip operation
    xmin = max(xmin, axmin)
    xmax = min(xmax, axmax)
    
    # transform xmin, xmax to cx and w
    cx = (xmin + xmax) / 2.0;
    w = xmax - xmin
    bbox = bbox.copy()
    bbox[0:4] = [cx, cy, w, h]
    return bbox

def rotate_horizontal_bbox_to_oriented(center, bbox):
    """
    Step 4 of Figure 5 in seglink paper: 
        Rotate the cropped horizontal bbox back to its original direction
    Args:
        center: the center of rotation
        bbox: [cx, cy, w, h, theta]
    Return: the oriented bbox
    """
    assert np.shape(center) == (2, ), "center must be a vector of length 2"
    assert np.shape(bbox) == (5, ) , "bbox must be a vector of length 4 or 5"
    bbox = np.asarray(bbox.copy(), dtype = np.float32)
    
    cx, cy, w, h, theta = bbox;
    M = cv2.getRotationMatrix2D(center, -theta, scale = 1) # 2x3
    cx, cy = np.dot(M, np.transpose([cx, cy, 1]))
    bbox[0:2] = [cx, cy]
    return bbox


def cal_seg_loc_for_single_anchor(anchor, rect):
    """
    Step 2 to 4
    """
    # rotate text box along the center of anchor to horizontal direction
    center = (anchor[0], anchor[1])
    rect = rotate_oriented_bbox_to_horizontal(center, rect)

    # crop horizontal text box to anchor    
    rect = crop_horizontal_bbox_using_anchor(rect, anchor)
    
    # rotate the box to original direction
    rect = rotate_horizontal_bbox_to_oriented(center, rect)
    
    return rect    
    

@util.dec.print_calling_in_short_for_tf
def match_anchor_to_text_boxes(anchors, xs, ys):
    """Match anchors to text boxes. 
       Return:
           seg_labels: shape = (N,), the seg_labels of segments. each value is the index of matched box if >=0.  
           seg_locations: shape = (N, 5), the absolute location of segments. Only the match segments are correctly calculated.
           
    """
    
    assert len(np.shape(anchors)) == 2 and np.shape(anchors)[1] == 4, "the anchors must be a tensor with shape = (num_anchors, 4)"
    assert len(np.shape(xs)) == 2 and np.shape(xs) == np.shape(ys) and np.shape(ys)[1] == 4, "the xs, ys must be a tensor with shape = (num_bboxes, 4)"
    anchors = np.asarray(anchors, dtype = np.float32)
    xs = np.asarray(xs, dtype = np.float32)
    ys = np.asarray(ys, dtype = np.float32)
    
    num_anchors = anchors.shape[0]
    seg_labels = np.ones((num_anchors, ), dtype = np.int32) * -1;
    seg_locations = np.zeros((num_anchors, 5), dtype = np.float32)
    
    # to avoid ln(0) in the ending process later.
    #     because the height and width will be encoded using ln(w_seg / w_anchor)
    seg_locations[:, 2] = anchors[:, 2]
    seg_locations[:, 3] = anchors[:, 3]
    
    num_bboxes = xs.shape[0]
    
    
    #represent bboxes using min area rects
    rects = min_area_rect(xs, ys) # shape = (num_bboxes, 5)
    rects = transform_cv_rect(rects)
    assert rects.shape == (num_bboxes, 5)
    
    #represent bboxes using contours
    cnts = []
    for bbox_idx in xrange(num_bboxes):
        bbox_points = zip(xs[bbox_idx, :], ys[bbox_idx, :])
        cnt = util.img.points_to_contour(bbox_points);
        cnts.append(cnt)
        
    import time
    start_time = time.time()
    # match anchor to bbox
    for anchor_idx in xrange(num_anchors):
        anchor = anchors[anchor_idx, :]
        acx, acy, aw, ah = anchor
        center_point_matched = False
        height_matched = False
        for bbox_idx in xrange(num_bboxes):
            # center point check
            center_point_matched = util.img.is_in_contour((acx, acy), cnts[bbox_idx])
            if not center_point_matched:
                continue
                
            # height height_ratio check
            rect = rects[bbox_idx, :]
            height_ratio = anchor_rect_height_ratio(anchor, rect)
            height_matched = height_ratio <= config.max_height_ratio
            if height_matched and center_point_matched:
                # an anchor can only be matched to at most one bbox
                seg_labels[anchor_idx] = bbox_idx
                seg_locations[anchor_idx, :] = cal_seg_loc_for_single_anchor(anchor, rect)
        
    end_time = time.time()
    tf.logging.info('Time in For Loop: %f'%(end_time - start_time))
    return seg_labels, seg_locations

# @util.dec.print_calling_in_short_for_tf
def match_anchor_to_text_boxes_fast(anchors, xs, ys):
    """Match anchors to text boxes. 
       Return:
           seg_labels: shape = (N,), the seg_labels of segments. each value is the index of matched box if >=0.  
           seg_locations: shape = (N, 5), the absolute location of segments. Only the match segments are correctly calculated.
           
    """
    
    assert len(np.shape(anchors)) == 2 and np.shape(anchors)[1] == 4, "the anchors must be a tensor with shape = (num_anchors, 4)"
    assert len(np.shape(xs)) == 2 and np.shape(xs) == np.shape(ys) and np.shape(ys)[1] == 4, "the xs, ys must be a tensor with shape = (num_bboxes, 4)"
    anchors = np.asarray(anchors, dtype = np.float32)
    xs = np.asarray(xs, dtype = np.float32)
    ys = np.asarray(ys, dtype = np.float32)
    
    num_anchors = anchors.shape[0]
    seg_labels = np.ones((num_anchors, ), dtype = np.int32) * -1;
    seg_locations = np.zeros((num_anchors, 5), dtype = np.float32)
    
    # to avoid ln(0) in the ending process later.
    #     because the height and width will be encoded using ln(w_seg / w_anchor)
    seg_locations[:, 2] = anchors[:, 2]
    seg_locations[:, 3] = anchors[:, 3]
    
    num_bboxes = xs.shape[0]
    
    
    #represent bboxes using min area rects
    rects = min_area_rect(xs, ys) # shape = (num_bboxes, 5)
    rects = transform_cv_rect(rects)
    assert rects.shape == (num_bboxes, 5)
    
    # construct a bbox point map: keys are the poistion of all points in bbox contours, and 
    #    value being the bbox index
    bbox_mask = np.ones(config.image_shape, dtype = np.int32) * (-1)
    for bbox_idx in xrange(num_bboxes):
        bbox_points = zip(xs[bbox_idx, :], ys[bbox_idx, :])
        bbox_cnts = util.img.points_to_contours(bbox_points)
        util.img.draw_contours(bbox_mask, bbox_cnts, -1, color = bbox_idx, border_width = - 1)
    
    points_in_bbox_mask = np.where(bbox_mask >= 0)
    points_in_bbox_mask = set(zip(*points_in_bbox_mask))
    points_in_bbox_mask = points_in_bbox_mask.intersection(config.default_anchor_center_set)
    
    for point in points_in_bbox_mask:
        anchors_here = config.default_anchor_map[point]
        for anchor_idx in anchors_here:
            anchor = anchors[anchor_idx, :]
            bbox_idx = bbox_mask[point]
            acx, acy, aw, ah = anchor
            height_matched = False
                    
            # height height_ratio check
            rect = rects[bbox_idx, :]
            height_ratio = anchor_rect_height_ratio(anchor, rect)
            height_matched = height_ratio <= config.max_height_ratio
            if height_matched:
                # an anchor can only be matched to at most one bbox
                seg_labels[anchor_idx] = bbox_idx
                seg_locations[anchor_idx, :] = cal_seg_loc_for_single_anchor(anchor, rect)
    return seg_labels, seg_locations


############################################################################################################
#                       link_gt calculation                                                                #
############################################################################################################
def reshape_link_gt_by_layer(link_gt):
    inter_layer_link_gts = {}
    cross_layer_link_gts = {}
    
    idx = 0;
    for layer_idx, layer_name in enumerate(config.feat_layers):
        layer_shape = config.feat_shapes[layer_name]
        lh, lw = layer_shape
        
        length = lh * lw * 8;
        layer_link_gt = link_gt[idx: idx + length]
        idx = idx + length;
        layer_link_gt = np.reshape(layer_link_gt, (lh, lw, 8))
        inter_layer_link_gts[layer_name] = layer_link_gt
        
    for layer_idx in xrange(1, len(config.feat_layers)):
        layer_name = config.feat_layers[layer_idx]
        layer_shape = config.feat_shapes[layer_name]
        lh, lw = layer_shape
        length = lh * lw * 4;
        layer_link_gt = link_gt[idx: idx + length]
        idx = idx + length;
        layer_link_gt = np.reshape(layer_link_gt, (lh, lw, 4))
        cross_layer_link_gts[layer_name] = layer_link_gt
    
    assert idx == len(link_gt)
    return inter_layer_link_gts, cross_layer_link_gts
        
def reshape_labels_by_layer(labels):
    layer_labels = {}
    idx = 0;
    for layer_name in config.feat_layers:
        layer_shape = config.feat_shapes[layer_name]
        label_length = np.prod(layer_shape)
        
        layer_match_result = labels[idx: idx + label_length]
        idx = idx + label_length;
        
        layer_match_result = np.reshape(layer_match_result, layer_shape)
        
        layer_labels[layer_name] = layer_match_result;
    assert idx == len(labels)
    return layer_labels;

def get_inter_layer_neighbours(x, y):
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
            (x - 1, y),                 (x + 1, y),  \
            (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]
    
def get_cross_layer_neighbours(x, y):
    return [(2 * x, 2 * y), (2 * x + 1, 2 * y), (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)]
    
def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h;

def cal_link_labels(labels):
    layer_labels = reshape_labels_by_layer(labels)
    inter_layer_link_gts = []
    cross_layer_link_gts = []
    for layer_idx, layer_name in enumerate(config.feat_layers):
        layer_match_result = layer_labels[layer_name]
        h, w = config.feat_shapes[layer_name]
        
        # initalize link groundtruth for the current layer
        inter_layer_link_gt = np.ones((h, w, 8), dtype = np.int32) * (-1)
        
        if layer_idx > 0: # no cross-layer link for the first layer. 
            cross_layer_link_gt = np.ones((h, w, 4), dtype = np.int32) * (-1)
            
        for x in xrange(w):
            for y in xrange(h):
                # the value in layer_match_result stands for the bbox idx a segments matches 
                # if less than 0, not matched.
                # only matched segments are considered in link_gt calculation
                if layer_match_result[y, x] >= 0:
                    matched_idx = layer_match_result[y, x]
                    
                    
                    # inter-layer link_gt calculation
                    # calculate inter-layer link_gt using the bbox matching result of inter-layer neighbours 
                    neighbours = get_inter_layer_neighbours(x, y)
                    for nidx, nxy in enumerate(neighbours): # n here is short for neighbour
                        nx, ny = nxy
                        if is_valid_cord(nx, ny, w, h):
                            n_matched_idx = layer_match_result[ny, nx]
                            # if the current default box has matched the same bbox with this neighbour, \
                            # the linkage connecting them is labeled as positive.
                            if matched_idx == n_matched_idx: 
                                inter_layer_link_gt[y, x, nidx] = n_matched_idx;
                                
                    # cross layer link_gt calculation
                    if layer_idx > 0:
                        previous_layer_name = config.feat_layers[layer_idx - 1];
                        ph, pw = config.feat_shapes[previous_layer_name]
                        previous_layer_match_result = layer_labels[previous_layer_name]
                        neighbours = get_cross_layer_neighbours(x, y)
                        for nidx, nxy in enumerate(neighbours):
                            nx, ny = nxy
                            if is_valid_cord(nx, ny, pw, ph):
                                n_matched_idx = previous_layer_match_result[ny, nx]
                                if matched_idx == n_matched_idx:
                                    cross_layer_link_gt[y, x, nidx] = n_matched_idx;                             
                    
        inter_layer_link_gts.append(inter_layer_link_gt)
        
        if layer_idx > 0:
            cross_layer_link_gts.append(cross_layer_link_gt)
    
    # construct the final link_gt from layer-wise data.
    # note that this reshape and concat order is the same with that of predicted linkages, which\
    #     has been done in the construction of SegLinkNet.
    inter_layer_link_gts = np.hstack([np.reshape(t, -1) for t in inter_layer_link_gts]);
    cross_layer_link_gts = np.hstack([np.reshape(t, -1) for t in cross_layer_link_gts]);
    link_gt = np.hstack([inter_layer_link_gts, cross_layer_link_gts])
    return link_gt

# @util.dec.print_calling_in_short_for_tf
def encode_seg_offsets(seg_locs):
    """
    Args:
        seg_locs: a ndarray with shape = (N, 5). It contains the abolute values of segment locations 
    Return:
        seg_offsets, i.e., the offsets from default boxes. It is used as the final segment location ground truth.
    """
    anchors = config.default_anchors
    anchor_cx, anchor_cy, anchor_w, anchor_h = (anchors[:, idx] for idx in range(4))
    seg_cx, seg_cy, seg_w, seg_h = (seg_locs[:, idx] for idx in range(4))
    
    #encoding using the formulations from Euqation (2) to (6) of seglink paper
    #    seg_cx = anchor_cx + anchor_w * offset_cx
    offset_cx = (seg_cx - anchor_cx) * 1.0 / anchor_w
    
    #    seg_cy = anchor_cy + anchor_w * offset_cy
    offset_cy = (seg_cy - anchor_cy) * 1.0 / anchor_h
    
    #    seg_w = anchor_w * e^(offset_w)
    offset_w = np.log(seg_w * 1.0 / anchor_w)
    #    seg_h = anchor_w * e^(offset_h)
    offset_h = np.log(seg_h * 1.0 / anchor_h)
    
    # prior scaling can be used to adjust the loss weight of loss on offset x, y, w, h, theta
    seg_offsets = np.zeros_like(seg_locs)
    seg_offsets[:, 0] = offset_cx / config.prior_scaling[0]
    seg_offsets[:, 1] = offset_cy / config.prior_scaling[1]
    seg_offsets[:, 2] = offset_w / config.prior_scaling[2]
    seg_offsets[:, 3] = offset_h / config.prior_scaling[3]
    seg_offsets[:, 4] = seg_locs[:, 4]  / config.prior_scaling[4]
    return seg_offsets

def decode_seg_offsets_pred(seg_offsets_pred):
    anchors = config.default_anchors
    anchor_cx, anchor_cy, anchor_w, anchor_h = (anchors[:, idx] for idx in range(4))
    
    offset_cx = seg_offsets_pred[:, 0] * config.prior_scaling[0]
    offset_cy = seg_offsets_pred[:, 1] * config.prior_scaling[1]
    offset_w = seg_offsets_pred[:, 2]  * config.prior_scaling[2] 
    offset_h = seg_offsets_pred[:, 3]  * config.prior_scaling[3]
    offset_theta = seg_offsets_pred[:, 4] * config.prior_scaling[4]
    
    seg_cx = anchor_cx + anchor_w * offset_cx
    seg_cy = anchor_cy + anchor_h * offset_cy # anchor_h == anchor_w
    seg_w = anchor_w * np.exp(offset_w)
    seg_h = anchor_h * np.exp(offset_h)
    seg_theta = offset_theta
    
    seg_loc = np.transpose(np.vstack([seg_cx, seg_cy, seg_w, seg_h, seg_theta]))
    return seg_loc

# @util.dec.print_calling_in_short_for_tf
def get_all_seglink_gt(xs, ys, ignored):
    
    # calculate ground truths. 
    # for matching results, i.e., seg_labels and link_labels, the values stands for the 
    #     index of matched bbox
    assert len(np.shape(xs)) == 2 and \
            np.shape(xs)[-1] == 4 and \
            np.shape(ys) == np.shape(xs), \
        'the shape of xs and ys must be (N, 4), but got %s and %s'%(np.shape(xs), np.shape(ys))
    
    assert len(xs) == len(ignored), 'the length of xs and `ignored` must be the same, \
            but got %s and %s'%(len(xs), len(ignored))
            
    anchors = config.default_anchors
    seg_labels, seg_locations = match_anchor_to_text_boxes_fast(anchors, xs, ys);
    link_labels = cal_link_labels(seg_labels)
    seg_offsets = encode_seg_offsets(seg_locations)
    
    
    # deal with ignored: use -2 to denotes ignored matchings temporarily
    def set_ignored_labels(labels, idx):
        cords = np.where(labels == idx)
        labels[cords] = -2
    
    ignored_bbox_idxes = np.where(ignored == 1)[0]
    for ignored_bbox_idx in ignored_bbox_idxes:
        set_ignored_labels(link_labels, ignored_bbox_idx)
        set_ignored_labels(seg_labels, ignored_bbox_idx)
        
        
    # deal with bbox idxes: use 1 to replace all matched label
    def set_positive_labels_to_one(labels):
        cords = np.where(labels >= 0)
        labels[cords] = 1
        
    set_positive_labels_to_one(seg_labels)
    set_positive_labels_to_one(link_labels)

    # deal with ignored: use 0 to replace all -2
    def set_ignored_labels_to_zero(labels):
        cords = np.where(labels == -2)
        labels[cords] = 0

    set_ignored_labels_to_zero(seg_labels)
    set_ignored_labels_to_zero(link_labels)

    # set dtypes    
    seg_labels = np.asarray(seg_labels, dtype = np.int32)
    seg_offsets = np.asarray(seg_offsets, dtype = np.float32)
    link_labels = np.asarray(link_labels, dtype = np.int32)
    
    return seg_labels, seg_offsets, link_labels
    

def tf_get_all_seglink_gt(xs, ys, ignored):
    """
    xs, ys: tensors reprensenting ground truth bbox, both with shape=(N, 4), values in 0~1
    """
    h_I, w_I = config.image_shape
    
    xs = xs * w_I
    ys = ys * h_I    
    seg_labels, seg_offsets, link_labels = tf.py_func(get_all_seglink_gt, [xs, ys, ignored], [tf.int32, tf.float32, tf.int32]);
    seg_labels.set_shape([config.num_anchors])
    seg_offsets.set_shape([config.num_anchors, 5])
    link_labels.set_shape([config.num_links])
    return seg_labels, seg_offsets, link_labels;

############################################################################################################
#                       linking segments together                                                          #
############################################################################################################
def group_segs(seg_scores, link_scores, seg_conf_threshold, link_conf_threshold):
    """
    group segments based on their scores and links.
    Return: segment groups as a list, consisting of list of segment indexes, reprensting a group of segments belonging to a same bbox.
    """
    
    assert len(np.shape(seg_scores)) == 1
    assert len(np.shape(link_scores)) == 1
    
    valid_segs = np.where(seg_scores >= seg_conf_threshold)[0];# `np.where` returns a tuple
    assert valid_segs.ndim == 1
    mask = {}
    for s in valid_segs:
        mask[s] = -1;
    
    def get_root(idx):
        parent = mask[idx]
        while parent != -1:
            idx = parent
            parent = mask[parent]
        return idx
            
    def union(idx1, idx2):
        root1 = get_root(idx1)
        root2 = get_root(idx2)
        
        if root1 != root2:
            mask[root1] = root2
            
    def to_list():
        result = {}
        for idx in mask:
            root = get_root(idx)
            if root not in result:
                result[root] = []
            
            result[root].append(idx)
            
        return [result[root] for root in result]

        
    seg_indexes = np.arange(len(seg_scores))
    layer_seg_indexes = reshape_labels_by_layer(seg_indexes)

    layer_inter_link_scores, layer_cross_link_scores = reshape_link_gt_by_layer(link_scores)
    
    for layer_index, layer_name in enumerate(config.feat_layers):
        layer_shape = config.feat_shapes[layer_name]
        lh, lw = layer_shape
        layer_seg_index = layer_seg_indexes[layer_name]
        layer_inter_link_score = layer_inter_link_scores[layer_name]
        if layer_index > 0:
            previous_layer_name = config.feat_layers[layer_index - 1]
            previous_layer_seg_index = layer_seg_indexes[previous_layer_name]
            previous_layer_shape = config.feat_shapes[previous_layer_name]
            plh, plw = previous_layer_shape
            layer_cross_link_score = layer_cross_link_scores[layer_name]
            
            
        for y in xrange(lh):
            for x in xrange(lw):
                seg_index = layer_seg_index[y, x]
                _seg_score = seg_scores[seg_index]
                if _seg_score >= seg_conf_threshold:

                    # find inter layer linked neighbours                    
                    inter_layer_neighbours = get_inter_layer_neighbours(x, y)
                    for nidx, nxy in enumerate(inter_layer_neighbours):
                        nx, ny = nxy
                        
                        # the condition of connecting neighbour segment: valid coordinate, 
                        # valid segment confidence and valid link confidence.
                        if is_valid_cord(nx, ny, lw, lh) and \
                            seg_scores[layer_seg_index[ny, nx]]  >= seg_conf_threshold and \
                            layer_inter_link_score[y, x, nidx] >= link_conf_threshold:
                            n_seg_index = layer_seg_index[ny, nx]
                            union(seg_index, n_seg_index)
                    
                    # find cross layer linked neighbours
                    if layer_index > 0:
                        cross_layer_neighbours = get_cross_layer_neighbours(x, y)
                        for nidx, nxy in enumerate(cross_layer_neighbours):
                            nx, ny = nxy
                            if is_valid_cord(nx, ny, plw, plh) and \
                               seg_scores[previous_layer_seg_index[ny, nx]]  >= seg_conf_threshold and \
                               layer_cross_link_score[y, x, nidx] >= link_conf_threshold:
                               
                                n_seg_index = previous_layer_seg_index[ny, nx]
                                union(seg_index, n_seg_index)

    return to_list()
        
        
    
############################################################################################################
#                       combining segments to bboxes                                                       #
############################################################################################################
def tf_seglink_to_bbox(seg_cls_pred, link_cls_pred, seg_offsets_pred, image_shape, 
                       seg_conf_threshold = None, link_conf_threshold = None):
    if len(seg_cls_pred.shape) == 3:
        assert seg_cls_pred.shape[0] == 1 # only batch_size == 1 supported now TODO
        seg_cls_pred = seg_cls_pred[0, ...]
        link_cls_pred = link_cls_pred[0, ...]
        seg_offsets_pred = seg_offsets_pred[0, ...]
        image_shape = image_shape[0, :]
    
    assert seg_cls_pred.shape[-1] == 2
    assert link_cls_pred.shape[-1] == 2
    assert seg_offsets_pred.shape[-1] == 5
    
    seg_scores = seg_cls_pred[:, 1]
    link_scores = link_cls_pred[:, 1]
    image_bboxes = tf.py_func(seglink_to_bbox, 
          [seg_scores, link_scores, seg_offsets_pred, image_shape, seg_conf_threshold, link_conf_threshold], 
          tf.float32);
    return image_bboxes
    
    
def seglink_to_bbox(seg_scores, link_scores, seg_offsets_pred, 
                    image_shape = None, seg_conf_threshold = None, link_conf_threshold = None):
    """
    Args:
        seg_scores: the scores of segments being positive
        link_scores: the scores of linkage being positive
        seg_offsets_pred
    Return:
        bboxes, with shape = (N, 5), and N is the number of predicted bboxes
    """
    seg_conf_threshold = seg_conf_threshold or config.seg_conf_threshold
    link_conf_threshold = link_conf_threshold or config.link_conf_threshold
    if image_shape is None:
        image_shape = config.image_shape

    seg_groups = group_segs(seg_scores, link_scores, seg_conf_threshold, link_conf_threshold);
    seg_locs = decode_seg_offsets_pred(seg_offsets_pred)
    
    bboxes = []
    ref_h, ref_w = config.image_shape
    for group in seg_groups:
        group = [seg_locs[idx, :] for idx in group]
        bbox = combine_segs(group)
        image_h, image_w = image_shape[0:2]
        scale = [image_w * 1.0 / ref_w, image_h * 1.0 / ref_h, image_w * 1.0 / ref_w, image_h * 1.0 / ref_h, 1]
        bbox = np.asarray(bbox) * scale
        bboxes.append(bbox)
        
    bboxes = bboxes_to_xys(bboxes, image_shape)
    return np.asarray(bboxes, dtype = np.float32)

def sin(theta):
    return np.sin(theta / 180.0 * np.pi)
def cos(theta):
    return np.cos(theta / 180.0 *  np.pi)
def tan(theta):
    return np.tan(theta / 180.0 * np.pi)
    
def combine_segs(segs, return_bias = False):
    segs = np.asarray(segs)
    assert segs.ndim == 2
    assert segs.shape[-1] == 5    
    
    if len(segs) == 1:
        return segs[0, :]
    
    # find the best straight line fitting all center points: y = kx + b
    cxs = segs[:, 0]
    cys = segs[:, 1]

    ## the slope
    bar_theta = np.mean(segs[:, 4])# average theta
    k = tan(bar_theta);
    
    ## the bias: minimize sum (k*x_i + b - y_i)^2
    ### let c_i = k*x_i - y_i
    ### sum (k*x_i + b - y_i)^2 = sum(c_i + b)^2
    ###                           = sum(c_i^2 + b^2 + 2 * c_i * b)
    ###                           = n * b^2 + 2* sum(c_i) * b + sum(c_i^2)
    ### the target b = - sum(c_i) / n = - mean(c_i) = mean(y_i - k * x_i)
    b = np.mean(cys - k * cxs)
    
    # find the projections of all centers on the straight line
    ## firstly, move both the line and centers upward by distance b, so as to make the straight line crossing the point(0, 0): y = kx
    ## reprensent the line as a vector (1, k), and the projection of vector(x, y) on (1, k) is: proj = (x + k * y)  / sqrt(1 + k^2)
    ## the projection point of (x, y) on (1, k) is (proj * cos(theta), proj * sin(theta))
    t_cys = cys - b
    projs = (cxs + k * t_cys) / np.sqrt(1 + k**2)
    proj_points = np.transpose([projs * cos(bar_theta), projs * sin(bar_theta)])
    
    # find the max distance
    max_dist = -1;
    idx1 = -1;
    idx2 = -1;

    for i in xrange(len(proj_points)):
        point1 = proj_points[i, :]
        for j in xrange(i + 1, len(proj_points)):
            point2 = proj_points[j, :]
            dist = np.sqrt(np.sum((point1 - point2) ** 2))
            if dist > max_dist:
                idx1 = i
                idx2 = j
                max_dist = dist
    assert idx1 >= 0 and idx2 >= 0
    # the bbox: bcx, bcy, bw, bh, average_theta
    seg1 = segs[idx1, :]
    seg2 = segs[idx2, :]
    bcx, bcy = (seg1[:2] + seg2[:2]) / 2.0
    bh = np.mean(segs[:, 3])
    bw = max_dist + (seg1[2] + seg2[2]) / 2.0
    
    if return_bias:
        return bcx, bcy, bw, bh, bar_theta, b# bias is useful for debugging.
    else:
        return bcx, bcy, bw, bh, bar_theta
            
def bboxes_to_xys(bboxes, image_shape):
    """Convert Seglink bboxes to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    if len(bboxes) == 0:
        return []
    
    assert np.ndim(bboxes) == 2 and np.shape(bboxes)[-1] == 5, 'invalid `bboxes` param with shape =  ' + str(np.shape(bboxes))
    
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x
    
    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y
    
    xys = np.zeros((len(bboxes), 8))
    for bbox_idx, bbox in enumerate(bboxes):
        bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox[4])
        points = cv2.cv.BoxPoints(bbox)
        points = np.int0(points)
        for i_xy, (x, y) in enumerate(points):
            x = get_valid_x(x)
            y = get_valid_y(y)
            points[i_xy, :] = [x, y]
        points = np.reshape(points, -1)
        xys[bbox_idx, :] = points
    return xys