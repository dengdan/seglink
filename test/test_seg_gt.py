import numpy as np
import cv2
from tf_extended.bboxes import  min_area_rect, transform_cv_rect, rotate_oriented_bbox_to_horizontal, crop_horizontal_bbox_using_anchor, rotate_horizontal_bbox_to_oriented
import util


def draw_points_as_contour(mask, points):
    cnts = util.img.points_to_contours(points);
    util.img.draw_contours(mask, cnts, color = util.img.COLOR_WHITE)

def get_min_area_rect_from_points(points):
    points = np.expand_dims(points, 0)
    xs = points[:, :, 0]
    ys = points[:, :, 1]
    
    rect = min_area_rect(xs, ys)[0, ...]
    return rect
    
def draw_horizontal_rect(mask, rect, text_pos = None, color = util.img.COLOR_GREEN):
    if text_pos is not None:
        util.img.put_text(mask, pos = text_pos, scale=0.5, text = 'trans: cx=%d, cy=%d, w=%d, h=%d, theta_0=%f'%(rect[0], rect[1], rect[2], rect[3], rect[4]))
    rect = np.asarray(rect, dtype = np.float32)
    cx, cy, w, h = rect[0:4]
    xmin = cx - w / 2
    xmax = cx + w / 2
    ymin = cy - h / 2
    ymax = cy + h / 2
    util.img.circle(mask, (cx, cy), 3, color = color)
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
    
def _test_min_area_rect():
    
#    points = [200, 200, 500, 200, 500, 300, 200, 300] # perpendicular
#    points = [200, 200, 300, 200, 300, 500, 200, 500] # perpendicular
#    points = [860,288,1147,180,1182,238,908,385] # right incline
    points = [573,238,733,357,733,397,573,277]#left inclined
    points = [546,382,564,368,574,471,556,484] # left inclined, nearly perpendicular
    points = [307,144,409,140,406,166,304,169]# right inclined, nearly perpendicular
    points = np.asarray(points, dtype = np.float32)
    points = np.reshape(points, (4, 2))
    wI = points.max() * 1.1
    mask = util.img.black((wI, wI, 3))
    draw_points_as_contour(mask, points)

    rect = get_min_area_rect_from_points(points)

    draw_oriented_rect(mask, rect, text_pos = (0, 100))
    util.sit(mask)

def _test_rotate_oriented_bbox_to_horizontal():
    points = [573,238,733,357,733,397,573,277] #left inclined
    points = np.asarray(points, dtype = np.float32)
    points = np.reshape(points, (4, 2))
    wI = points.max() * 1.1
    mask = util.img.black((wI, wI, 3))
    draw_points_as_contour(mask, points)
    
    rect = get_min_area_rect_from_points(points)
    draw_oriented_rect(mask, rect, text_pos = (0, 100))
    
    rect = transform_cv_rect(rect)[0, ...]
    
    rect = rotate_oriented_bbox_to_horizontal((rect[0], rect[1]), rect)
    draw_horizontal_rect(mask, rect, text_pos = (0, 130))
        
    util.sit(mask)
    
def _test_cal_seg_gt():
    points = [200, 200, 300, 200, 300, 500, 200, 500] # perpendicular
    points = [200, 200, 500, 200, 500, 300, 200, 300] # perpendicular
    points = [307,144,409,140,406,166,304,169]# right inclined, nearly perpendicular
    points = [573,238,733,357,733,397,573,277]#left inclined
    points = [860,288,1147,180,1182,238,908,385] # right inclined
    points = [546,382,564,368,574,471,556,484] # left inclined, nearly perpendicular

    points = np.asarray(points, dtype = np.float32)
    points = np.reshape(points, (4, 2))
    wI = points.max() * 1.1
    mask = util.img.black((wI, wI, 3))
    #draw_points_as_contour(mask, points)
    
    rect = get_min_area_rect_from_points(points)
    draw_oriented_rect(mask, rect, text_pos = (0, 100))
    
    # the text bbox represented as [cx, cy, w, h, theta]
    rect = transform_cv_rect(rect)[0, ...]
    
    # the anchor to be matched    
    ah = min(rect[3], rect[2])#rect[3]
    anchor = [rect[0] + 5, rect[1] + 5, ah * 1.2, ah * 1.2]
#    anchor = [rect[0] + 30, rect[1] + 5, ah * 0.8, ah * 0.8]
    center = (anchor[0], anchor[1])
    draw_horizontal_rect(mask, anchor, color = util.img.COLOR_RGB_YELLOW)
    
    # rotate text box along the center of anchor to horizontal direction
    rect = rotate_oriented_bbox_to_horizontal(center, rect)
    draw_horizontal_rect(mask, rect, color = util.img.COLOR_RGB_BLUE, text_pos = (0, 130))

    # crop horizontal text box to anchor    
    rect = crop_horizontal_bbox_using_anchor(rect, anchor)
    draw_horizontal_rect(mask, rect, color = util.img.COLOR_GREEN)
    
    # rotate the box to original direction
    rect = rotate_horizontal_bbox_to_oriented(center, rect)
    draw_oriented_rect(mask, rect, color = util.img.COLOR_RGB_PINK)
    util.sit(mask)
    return
if __name__ == '__main__':
    #_test_min_area_rect()
    _test_cal_seg_gt()
