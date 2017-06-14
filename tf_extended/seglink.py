import cv2
import numpy as np

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
    xs: numpy ndarray of shape N*4, [x1, x2, x3, x4]
    ys: numpy ndarray of shape N*4, [y1, y2, y3, y4]
    return the oriented rects sorrounding the box represented by [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
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
    

def transform_cv_rect(rects):
    """Transform the rects from opencv method minAreaRect to our rects. 
    Step 1 of Figure 5 in seglink paper
    rects: (5, ) or (N, 5)
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
        assert theta < 0 and theta >= -90, "invalid theta: %f"%(theta) 
        if abs(theta) > 45 or (abs(theta) == 45 and w < h):
            w, h = [h, w]
            theta = 90 + theta
        rects[idx, ...] = [cx, cy, w, h, theta]
    if only_one:
        return rects[0, ...]
    return rects                
    

def rotate_oriented_bbox_to_horizontal(center, bbox):
    """
    center: the center of rotation
    bbox: [cx, cy, w, h, theta]
    Step 2 of Figure 5 in seglink paper
    """
    assert np.shape(center) == (2, ), "center must be a vector of length 2"
    assert np.shape(bbox) == (5, ) or np.shape(bbox) == (4, ), "bbox must be a vector of length 4 or 5"
    bbox = np.asarray(bbox.copy(), dtype = np.float32)
    
    cx, cy, w, h, theta = bbox[...];
    M = cv2.getRotationMatrix2D(center, theta, scale = 1) # 2x3
    
    cx, cy = np.dot(M, np.transpose([cx, cy, 1]))
    
    bbox[0:2] = [cx, cy]
    return bbox

def crop_horizontal_bbox_using_anchor(bbox, anchor):
    """Step 3 in Figure 5 in seglink paper
    """
    assert np.shape(anchor) == (4, ), "anchor must be a vector of length 4"
    assert np.shape(bbox) == (5, ) or np.shape(bbox) == (4, ), "bbox must be a vector of length 4 or 5"
    acx, acy, aw, ah = anchor
    axmin = acx - aw / 2.0;
    axmax = acx + aw / 2.0;
    
    cx, cy, w, h = bbox[0:4]
    xmin = cx - w / 2.0
    xmax = cx + w / 2.0
    
    xmin = max(xmin, axmin)
    xmax = min(xmax, axmax)
    
    cx = (xmin + xmax) / 2.0;
    w = xmax - xmin
    bbox = bbox.copy()
    bbox[0:4] = [cx, cy, w, h]
    return bbox

def rotate_horizontal_bbox_to_oriented(center, bbox):
    """
    center: the center of rotation
    bbox: [cx, cy, w, h, theta]
    Step 4 of Figure 5 in seglink paper
    """
    assert np.shape(center) == (2, ), "center must be a vector of length 2"
    assert np.shape(bbox) == (5, ) or np.shape(bbox) == (4, ), "bbox must be a vector of length 4 or 5"
    bbox = np.asarray(bbox.copy(), dtype = np.float32)
    
    cx, cy, w, h, theta = bbox[...];
    M = cv2.getRotationMatrix2D(center, -theta, scale = 1) # 2x3
    cx, cy = np.dot(M, np.transpose([cx, cy, 1]))
    bbox[0:2] = [cx, cy]
    return bbox


def cal_seg_gt_for_single_anchor(anchor, rect):
    # rotate text box along the center of anchor to horizontal direction
    center = (anchor[0], anchor[1])
    rect = rotate_oriented_bbox_to_horizontal(center, rect)

    # crop horizontal text box to anchor    
    rect = crop_horizontal_bbox_using_anchor(rect, anchor)
    
    # rotate the box to original direction
    rect = rotate_horizontal_bbox_to_oriented(center, rect)
    return rect    
    

def match_anchor_to_text_boxes(anchors, xs, ys):
    """Match anchors to text boxes. 
       The match results are stored in a vector, each of whose is the index of matched box if >=0, and returned.
    """
    
    assert len(np.shape(anchors)) == 2 and np.shape(anchors)[1] == 4, "the anchors must be a tensor with shape = (num_anchors, 4)"
    assert len(np.shape(xs)) == 2 and np.shape(xs) == np.shape(ys) and np.shape(ys)[1] == 4, "the xs, ys must be a tensor with shape = (num_bboxes, 4)"
    anchors = np.asarray(anchors, dtype = np.float32)
    xs = np.asarray(xs, dtype = np.float32)
    ys = np.asarray(ys, dtype = np.float32)
    
    num_anchors = anchors.shape[0]
    labels = np.ones((num_anchors, ), dtype = np.int32) * -1;
    seg_gt = np.zeros((num_anchors, 5), dtype = np.float32)
    num_bboxes = xs.shape[0]
    
    #represent bboxes with min area rects
    rects = min_area_rect(xs, ys) # shape = (num_bboxes, 5)
    rects = transform_cv_rect(rects)
    assert rects.shape == (num_bboxes, 5)
    
    #represent bboxes with contours
    cnts = []
    for bbox_idx in xrange(num_bboxes):
        bbox_points = zip(xs[bbox_idx, :], ys[bbox_idx, :])
        cnt = util.img.points_to_contour(bbox_points);
        cnts.append(cnt)
    # match
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
                
            # height ratio check
            rect = rects[bbox_idx, :]
            cx, cy, w, h = rect[0:4]
            height = min(w, h);
            ratio = aw / height # aw == ah
            height_matched = max(ratio, 1/ratio) <= config.MATCHING_max_height_ratio
            
            if height_matched and center_point_matched:
                # an anchor can only be matched to at most one bbox
                labels[anchor_idx] = bbox_idx
                seg_gt[anchor_idx, :] = cal_seg_gt_for_single_anchor(anchor, rect)
                
    return labels, seg_gt



############################################################################################################
#                       link_gt calculation                                                                #
############################################################################################################

def reshape_link_gt_by_layer(link_gt, feat_layers, feat_shapes):
    inter_layer_link_gts = {}
    cross_layer_link_gts = {}
    
    idx = 0;
    for layer_idx, layer_name in enumerate(feat_layers):
        layer_shape = feat_shapes[layer_name]
        lh, lw = layer_shape
        
        length = lh * lw * 8;
        layer_link_gt = link_gt[idx: idx + length]
        idx = idx + length;
        layer_link_gt = np.reshape(layer_link_gt, (lh, lw, 8))
        inter_layer_link_gts[layer_name] = layer_link_gt
        
    for layer_idx in xrange(1, len(feat_layers)):
        layer_name = feat_layers[layer_idx]
        layer_shape = feat_shapes[layer_name]
        lh, lw = layer_shape
        length = lh * lw * 4;
        layer_link_gt = link_gt[idx: idx + length]
        idx = idx + length;
        layer_link_gt = np.reshape(layer_link_gt, (lh, lw, 4))
        cross_layer_link_gts[layer_name] = layer_link_gt
    
    assert idx == len(link_gt)
    return inter_layer_link_gts, cross_layer_link_gts
        
def reshape_labels_by_layer(labels, feat_layers, feat_shapes):
    layer_labels = {}
    idx = 0;
    for layer_name in feat_layers:
        layer_shape = feat_shapes[layer_name]
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
    return x >=0 and x < w and y >= 0 and y < h;

def cal_link_gt(labels, feat_layers, feat_shapes):
    layer_labels = reshape_labels_by_layer(labels, feat_layers, feat_shapes)
    inter_layer_link_gts = []
    cross_layer_link_gts = []
    for layer_idx, layer_name in enumerate(feat_layers):
        layer_match_result = layer_labels[layer_name]
        h, w = feat_shapes[layer_name]
        
        inter_layer_link_gt = np.zeros((h, w, 8), dtype = np.int32)
        
        if layer_idx > 0:
            cross_layer_link_gt = np.zeros((h, w, 4), dtype = np.int32)
            
        for x in xrange(w):
            for y in xrange(h):
                if layer_match_result[y, x] >= 0:
                    matched_idx = layer_match_result[y, x]
                    
                    # inter layer
                    neighbours = get_inter_layer_neighbours(x, y)
                    for nidx, nxy in enumerate(neighbours):
                        nx, ny = nxy
                        if is_valid_cord(nx, ny, w, h):
                            n_matched_idx = layer_match_result[ny, nx]
                            if matched_idx == n_matched_idx:
                                inter_layer_link_gt[y, x, nidx] = 1;
                                
                    # cross layer
                    if layer_idx > 0:
                        previous_layer_name = feat_layers[layer_idx - 1];
                        ph, pw = feat_shapes[previous_layer_name]
                        previous_layer_match_result = layer_labels[previous_layer_name]
                        neighbours = get_cross_layer_neighbours(x, y)
                        for nidx, nxy in enumerate(neighbours):
                            nx, ny = nxy
                            if is_valid_cord(nx, ny, pw, ph):
                                n_matched_idx = previous_layer_match_result[ny, nx]
                                if matched_idx == n_matched_idx:
                                    cross_layer_link_gt[y, x, nidx] = 1;                             
                    
        inter_layer_link_gts.append(inter_layer_link_gt)
        
        if layer_idx > 0:
            cross_layer_link_gts.append(cross_layer_link_gt)
    
    inter_layer_link_gts = np.hstack([np.reshape(t, -1) for t in inter_layer_link_gts]);
    cross_layer_link_gts = np.hstack([np.reshape(t, -1) for t in cross_layer_link_gts]);
    link_gt = np.hstack([inter_layer_link_gts, cross_layer_link_gts])
    return link_gt


def get_all_seglink_gt(anchors, xs, ys, feat_layers, feat_shapes):
    labels, seg_gt = match_anchor_to_text_boxes(anchors, xs, ys);
    link_gt = cal_link_gt(labels, feat_layers, feat_shapes);
    return labels, seg_gt, link_gt
    
    
