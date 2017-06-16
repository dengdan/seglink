import tensorflow as tf
import numpy as np
import cv2
from nets import seglink_symbol, anchor_layer
from tf_extended.seglink import  *
import util

    
def draw_horizontal_rect(mask, rect, text_pos = None, color = util.img.COLOR_GREEN, draw_center = True, center_only = False):
    if text_pos is not None:
        if len(rect) == 5:
            util.img.put_text(mask, pos = text_pos, scale=0.5, text = 'trans: cx=%d, cy=%d, w=%d, h=%d, theta_0=%f'%(rect[0], rect[1], rect[2], rect[3], rect[4]))
        else:
            util.img.put_text(mask, pos = text_pos, scale=0.5, text = 'trans: cx=%d, cy=%d, w=%d, h=%d, theta_0=0.0'%(rect[0], rect[1], rect[2], rect[3]))
    rect = np.asarray(rect, dtype = np.float32)
    cx, cy, w, h = rect[0:4]
    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2
    if draw_center or center_only:
        util.img.circle(mask, (cx, cy), 3, color = color)
    
    if not center_only:
        util.img.rectangle(mask, (xmin, ymin), (xmax, ymax), color = color)
    
    
    
def draw_oriented_rect(mask, rect, text_pos = None, color = util.img.COLOR_RGB_RED):
    if text_pos:
        util.img.put_text(mask, pos = text_pos, scale=0.5, text = 'cv2: cx=%d, cy=%d, w=%d, h=%d, theta_0=%f'%(rect[0], rect[1], rect[2], rect[3], rect[4]))
    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    box_center = rect[0]
    util.img.circle(mask, box_center, 3, color = color)
    cv2.drawContours(mask, [box], 0, color, 1)
    
    
def points_to_xys(points):
    points = np.asarray(points, dtype = np.float32)
    points = np.reshape(points, (-1, 4, 2))
    xs = points[..., 0]
    ys = points[..., 1]
    return xs, ys
    
    

def _test_matching_and_seg_gt_cal():

    points = [[200, 200, 300, 200, 300, 500, 200, 500]] # perpendicular
    points = [[200, 200, 500, 200, 500, 300, 200, 300]] # perpendicular
    points = [[307,144,409,140,406,166,304,169]]# right inclined, nearly perpendicular
    points = [[573,238,733,357,733,397,573,277]]#left inclined
    points = [[860,288,1147,180,1182,238,908,385]] # right inclined
    #points = [[546,382,564,368,574,471,556,484]] # left inclined, nearly perpendicular
    points = [[170,169,219,159,224,179,174,189], \
                [80,144,122,130,137,205,94,220], \
                [78,232,139,221,144,240,83,252], \
                [143,221,158,218,163,237,148,240], \
                [164,218,211,209,216,228,169,237], \
                [218,213,241,208,245,223,222,227], \
                [244,207,266,203,271,219,250,222]]
    xs, ys = points_to_xys(points)
    
    image_size = 2048
    fake_image = tf.ones((2, image_size, image_size, 3))
    feat_layers = ['conv4_3','fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2', 'conv10_2']
    fake_net = seglink_symbol.SegLinkNet(inputs = fake_image, feat_layers = feat_layers)
    shapes = fake_net.get_shapes();
    anchors, _ = anchor_layer.generate_anchors(image_shape = (image_size, image_size), feat_layers = feat_layers, feat_shapes = shapes)
    anchors = anchors * [image_size, image_size, image_size, image_size]
    labels, seg_gt = match_anchor_to_text_boxes(anchors, xs, ys)
    image = util.img.black((image_size, image_size, 3))
    
    min_area_rects = min_area_rect(xs, ys)
    seg_link_rects = transform_cv_rect(min_area_rects)
    
    for rect in min_area_rects:        
        draw_oriented_rect(image, rect, color=util.img.COLOR_WHITE)
        
    for anchor_idx, label in enumerate(labels):
        if label >= 0:
            anchor = anchors[anchor_idx, :]
            #draw_horizontal_rect(image, anchor, color = util.img.COLOR_RGB_YELLOW)

            rect = seg_gt[anchor_idx, :]
            #draw_horizontal_rect(image, rect)
            draw_oriented_rect(image, rect, color=util.img.COLOR_RGB_PINK)
    #util.sit(image)        
    util.plt.imwrite('~/temp/no-use/seg_gt/match.jpg', image)

def _test_cal_seglink_gt():
    points = [[200, 200, 300, 200, 300, 500, 200, 500]] # perpendicular
    points = [[200, 200, 500, 200, 500, 300, 200, 300]] # perpendicular
    points = [[307,144,409,140,406,166,304,169]]# right inclined, nearly perpendicular
    points = [[573,238,733,357,733,397,573,277]]#left inclined
    points = [[860,288,1147,180,1182,238,908,385]] # right inclined
    #points = [[546,382,564,368,574,471,556,484]] # left inclined, nearly perpendicular
    points = [[170,169,219,159,224,179,174,189], \
                [80,144,122,130,137,205,94,220], \
                [78,232,139,221,144,240,83,252], \
                [143,221,158,218,163,237,148,240], \
                [164,218,211,209,216,228,169,237], \
                [218,213,241,208,245,223,222,227], \
                [244,207,266,203,271,219,250,222]]
    xs, ys = points_to_xys(points)
    
    image_size = 2048
    fake_image = tf.ones((2, image_size, image_size, 3))
    feat_layers = ['conv4_3','fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2', 'conv10_2']
    fake_net = seglink_symbol.SegLinkNet(inputs = fake_image, feat_layers = feat_layers)
    feat_shapes = fake_net.get_shapes();
    anchors, layer_anchors = anchor_layer.generate_anchors(image_shape = (image_size, image_size), feat_layers = feat_layers, feat_shapes = feat_shapes)
    labels, seg_gt, link_gt = get_all_seglink_gt(anchors, xs, ys, feat_layers, feat_shapes)
    inter_layer_link_gt, cross_layer_link_gt = reshape_link_gt_by_layer(link_gt, feat_layers, feat_shapes)
    
    # test from here
    image = util.img.black((image_size, image_size, 3))
    min_area_rects = min_area_rect(xs, ys)
    for rect in min_area_rects:        
        draw_oriented_rect(image, rect, color=util.img.COLOR_WHITE)
    
    
    layer_labels = reshape_labels_by_layer(labels, feat_layers, feat_shapes)
    
    bbox_idx = 0
    for layer_idx, layer_name in enumerate(feat_layers):
        layer_anchor = layer_anchors[layer_name]
        layer_label = layer_labels[layer_name]
        layer_inter_link = inter_layer_link_gt[layer_name]
        
        layer_shape = feat_shapes[layer_name]
        lh, lw = layer_shape
        for x in xrange(lw):
            for y in xrange(lh):
                anchor = layer_anchor[y, x, :]
                anchor_label = layer_label[y, x]
                if anchor_label != bbox_idx:
                    continue
                
                img = image.copy()
                # draw all anchors matched the same box
                for anchor_idx, label in enumerate(labels):
                    if label == anchor_label:
                        _anchor = anchors[anchor_idx, :]
                        draw_horizontal_rect(img, _anchor, color = util.img.COLOR_RGB_YELLOW)
                
                # draw inter layer linked anchors
                anchor_inter_layer_link = layer_inter_link[y, x, :]
                assert len(anchor_inter_layer_link) == 8
                neighbours = get_inter_layer_neighbours(x, y)
                inter_neighbour_count = 0

                for nidx, nxy in enumerate(neighbours): 
                    nx, ny = nxy
                    if is_valid_cord(nx, ny, lw, lh):
                        if anchor_label == layer_label[ny, nx]:
                            inter_neighbour_count += 1
                            #draw the linked anchors
                            nanchor = layer_anchor[ny, nx, :]
                            #draw_horizontal_rect(img, nanchor, color = util.img.COLOR_GREEN)
                util.img.put_text(img, text = '(%d, %d): num_inter_layer_linkage = %d'%(x, y, inter_neighbour_count), pos = (0, 100))

                # draw cross layer linked anchors
                cross_neighbour_count = 0
                if layer_idx > 0:
                    previous_layer_anchor = layer_anchors[feat_layers[layer_idx - 1]]
                    previous_layer_label = layer_labels[feat_layers[layer_idx - 1]]
                    layer_cross_link = cross_layer_link_gt[layer_name]
                    anchor_cross_layer_link = layer_cross_link[y, x, :]
                    assert len(anchor_cross_layer_link) == 4
                    neighbours = get_cross_layer_neighbours(x, y)

                    for nidx, nxy in enumerate(neighbours): 
                        nx, ny = nxy
                        if is_valid_cord(nx, ny, lw, lh):
                            if anchor_label == previous_layer_label[ny, nx]:
                                cross_neighbour_count += 1
                                #draw the linked anchors
                                nanchor = previous_layer_anchor[ny, nx, :]
                                draw_horizontal_rect(img, nanchor, color = util.img.COLOR_GREEN)
                            
                util.img.put_text(img, text = '(%d, %d): num_cross_layer_linkage = %d'%(x, y, cross_neighbour_count), pos = (0, 130))
                    
                # draw current anchor
                draw_horizontal_rect(img, anchor, color = util.img.COLOR_RGB_RED)
                if cross_neighbour_count == 0:
                    continue
                    
                import pdb
                pdb.set_trace()
                # check the image, tell if it is right: No. green anchors == num_linkage
                util.sit(img)
        
def _test_combine_seglinks():
    points = [[200, 200, 300, 200, 300, 500, 200, 500]] # perpendicular
    points = [[200, 200, 500, 200, 500, 300, 200, 300]] # perpendicular
    points = [[307,144,409,140,406,166,304,169]]# right inclined, nearly perpendicular
    points = [[573,238,733,357,733,397,573,277]]#left inclined
    points = [[860,288,1147,180,1182,238,908,385]] # right inclined
    #points = [[546,382,564,368,574,471,556,484]] # left inclined, nearly perpendicular
    points = [[170,169,219,159,224,179,174,189], \
                [80,144,122,130,137,205,94,220], \
                [78,232,139,221,144,240,83,252], \
                [143,221,158,218,163,237,148,240], \
                [164,218,211,209,216,228,169,237], \
                [218,213,241,208,245,223,222,227], \
                [244,207,266,203,271,219,250,222]]
    xs, ys = points_to_xys(points)
    
    image_size = 2048
    fake_image = tf.ones((2, image_size, image_size, 3))
    feat_layers = ['conv4_3','fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2', 'conv10_2']
    fake_net = seglink_symbol.SegLinkNet(inputs = fake_image, feat_layers = feat_layers)
    feat_shapes = fake_net.get_shapes();
    anchors, layer_anchors = anchor_layer.generate_anchors(image_shape = (image_size, image_size), feat_layers = feat_layers, feat_shapes = feat_shapes)
    labels, seg_gt, link_gt = get_all_seglink_gt(anchors, xs, ys, feat_layers, feat_shapes)

    seg_scores = labels >= 0
    seg_groups = group_segs(seg_scores = seg_scores, link_scores = link_gt, feat_layers = feat_layers, feat_shapes = feat_shapes, seg_confidence_threshold = 0.5, link_confidence_threshold = 0.5)
        
    # test grouping
    image = util.img.black((image_size, image_size, 3))
    min_area_rects = min_area_rect(xs, ys)
    for rect in min_area_rects:        
        draw_oriented_rect(image, rect, color=util.img.COLOR_WHITE)

    #util.sit(img)
    
    def draw_line(I, theta, b, color):
        k = tan(theta)
        def fn(x):
            return int(k * x + b)
        h, w = I.shape[:-1]
        print 'theta = %f, y = %f * x + %f'%(theta, k, b)
        for x in xrange(w):
            yf = fn(x)
            
            if is_valid_cord(x, yf, w,h):
                I[yf, x, :] = color
    
    # test combining
    bboxes = seglink_to_bbox(seg_scores, link_scores = link_gt, segs = seg_gt, feat_layers = feat_layers, feat_shapes = feat_shapes, seg_confidence_threshold = 0.5, link_confidence_threshold = 0.5)
    
    img = image.copy()
    for group, bbox in zip(seg_groups, bboxes):
        for seg_idx in group:
            seg = seg_gt[seg_idx, :]
            #draw_oriented_rect(img, seg, color = util.img.COLOR_RGB_YELLOW)
        draw_oriented_rect(img, bbox, color = util.img.COLOR_GREEN)
        #draw_line(img, bbox[-2], bbox[-1], color = util.img.COLOR_RGB_RED)
    util.sit(img)
    print "check the output image, make sure that all white bboxes are overridden by green ones."
            
if __name__ == '__main__':
    #_test_min_area_rect()
#    _test_cal_seg_gt()
    #_test_matching_and_seg_gt_cal()
    _test_combine_seglinks()

