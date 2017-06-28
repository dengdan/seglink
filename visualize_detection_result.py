#encoding utf-8

import util

icdar2015_ch4_training_images = '~/dataset/ICDAR2015/Challenge4/ch4_training_images/'
icdar2015_ch4_training_gt = '~/dataset/ICDAR2015/Challenge4/ch4_training_localization_transcription_gt/'

training_image_name_pattern = icdar2015_ch4_training_images+'img_%d.jpg'
training_gt_name_pattern = icdar2015_ch4_training_gt+'gt_img_%d.txt'

def draw_bbox(image_data, line, color):
    line = util.str.remove_all(line, ',');
    data = gt.split();

    def draw_rectangle():
        x, y, w, h = (int(v) for v in data[0 : 4])    
        util.img.rectangle(
            img = image_data, 
            left_up = (x, y), 
            right_bottom = (x + w, y + h), 
            color = color, 
            border_width = 1)
        
    def draw_text():
        text = data[-1]
        pos = (int(v) for v in data[0:2])
        util.img.put_text(
            img = image_data, 
            text = text, 
            pos = pos, 
            scale = 1, 
            color = color)
    def draw_oriented_bbox():
        points = (int(v) for v in data[0:8])
        points = np.reshape(points, (4, 2))
        cnts = util.img.points_to_contours(points)
        util.img.draw_contours(img, cnts, -1, color = color, border_width = 1)
    
    if len(data) == 4: # ic13 det
        draw_rectangle();
    elif len(data) == 5: # ic13 gt
        draw_rectangle()
        draw_text()
    elif len(data) == 8:# ic15 det
        draw_oriented_bbox()
    elif len(data) == 9: # ic15 gt
        draw_oriented_bbox()
        draw_text()
        
       
def visiualize(image_root, det_root, output_root, gt_root = None):
    def read_gt_file(image_name):
        gt_file = util.io.join_path(gt_root, 'gt_%s.txt'%(image_name))
        return util.io.read_lines(gt_file)

    def read_det_file(image_name):
        det_file = util.io.join_path(det_root, 'res_%s.txt'%(image_name))
        return util.io.read_lines(det_file)
    
    def read_image_file(image_name):
        return util.io.imread(util.io.join_path(image_root, image_name))
    
    image_names = util.io.ls(image_root, '.jpg')
    for image_idx, image_name in enumerate(image_names):
        
        image_data = read_image_file(image_name) # in BGR
        
        image_name = util.io.get_filename(image_name)
        
        print '%d / %d: %s'%(image_idx + 1, len(image_names), image_name)
        
        det_image = image_data.copy()
        det_lines = read_det_file(image_name)
        for line in det_lines:
            draw_bbox(image_data, line, color = util.img.COLOR_BGR_RED)
        util.img.imwrite(util.io.join_path(output_root, '%s_pred.jpg'%(image_name)))
        
        if gt_root is not None:
            gt_lines = read_gt_file(image_name)
            for line in gt_lines:
                draw_bbox(image_data, line, color = util.img.COLOR_GREEN)
            util.img.imwrite(util.io.join_path(output_root, '%s_gt.jpg'%(image_name)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='visualize detection result of seglink')
    parser.add_argument('--image', type=str, required = True,help='the directory of test image')
    parser.add_argument('--gt', type=str, default=None,help='the directory of ground truth txt files')
    parser.add_argument('--det', type=str, required = True, help='the directory of detection result')
    parser.add_argument('--output', type=str, required = True, help='the directory to store images with bboxes')
    args = parser.parse_args()
    print('**************Arguments*****************')
    print(args)
    print('****************************************')
    visualize(image_root = args.image, gt_root = args.gt, det_root = args.det, output_root = args.output)
