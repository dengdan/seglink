#encoding=utf-8
import numpy as np;
import tensorflow as tf
import util;
from dataset_utils import int64_feature, float_feature, bytes_feature

# encoding = utf-8
import numpy as np    
import time

import util  


class SynthTextDataFetcher():
    def __init__(self, mat_path, root_path):
        self.mat_path = mat_path
        self.root_path = root_path
        self._load_mat()
        
    @util.dec.print_calling    
    def _load_mat(self):
        data = util.io.load_mat(self.mat_path)
        self.image_paths = data['imnames'][0]
        self.image_bbox = data['wordBB'][0]
        self.txts = data['txt'][0]
        self.num_images =  len(self.image_paths)

    def get_image_path(self, idx):
        image_path = util.io.join_path(self.root_path, self.image_paths[idx][0])
        return image_path

    def get_num_words(self, idx):
        try:
            return np.shape(self.image_bbox[idx])[2]
        except: # error caused by dataset
            return 1


    def get_word_bbox(self, img_idx, word_idx):
        boxes = self.image_bbox[img_idx]
        if len(np.shape(boxes)) ==2: # error caused by dataset
            boxes = np.reshape(boxes, (2, 4, 1))
             
        xys = boxes[:,:, word_idx]
        assert(np.shape(xys) ==(2, 4))
        return np.float32(xys)
    
    def normalize_bbox(self, xys, width, height):
        xs = xys[0, :]
        ys = xys[1, :]
        
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)
        
        # bound them in the valid range
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(width, max_x)
        max_y = min(height, max_y)
        
        # check the w, h and area of the rect
        w = max_x - min_x
        h = max_y - min_y
        is_valid = True
        
        if w < 10 or h < 10:
            is_valid = False
            
        if w * h < 100:
            is_valid = False
        
        xys[0, :] = xys[0, :] / width
        xys[1, :] = xys[1, :] / height
        
        return is_valid, min_x / width, min_y /height, max_x / width, max_y / height, xys
        
    def get_txt(self, image_idx, word_idx):
        txts = self.txts[image_idx];
        clean_txts = []
        for txt in txts:
            clean_txts += txt.split()
        return str(clean_txts[word_idx])
        
        
    def fetch_record(self, image_idx):
        image_path = self.get_image_path(image_idx)
        if not (util.io.exists(image_path)):
            return None;
        img = util.img.imread(image_path)
        h, w = img.shape[0:-1];
        num_words = self.get_num_words(image_idx)
        rect_bboxes = []
        full_bboxes = []
        txts = []
        for word_idx in xrange(num_words):
            xys = self.get_word_bbox(image_idx, word_idx);       
            is_valid, min_x, min_y, max_x, max_y, xys = self.normalize_bbox(xys, width = w, height = h)
            if not is_valid:
                continue;
            rect_bboxes.append([min_x, min_y, max_x, max_y])
            full_bboxes.append(xys);
            txt = self.get_txt(image_idx, word_idx);
            txts.append(txt);
        if len(rect_bboxes) == 0:
            return None;
        
        return image_path, img, txts, rect_bboxes, full_bboxes
    
        
def _convert_to_example(image_data, labels, labels_text, rect_bboxes, full_bboxes, shape, difficult, truncated):
    """Build an Example proto for an image example.
    Args:
      image_data: string, JPEG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of floats in [0, 1];
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    
    x1 = []; x2 = []; x3 = []; x4 = []
    y1 = []; y2 = []; y3 = []; y4 = []
    
    for xys in full_bboxes:
        xs = list(xys[0, :])
        ys = list(xys[1, :])
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([x1, x2, x3, x4], xs)]
        [l.append(point) for l, point in zip([y1, y2, y3, y4], ys)]
        # pylint: enable=expression-not-assigned
    
    
    for b in rect_bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
        # pylint: enable=expression-not-assigned
    image_format = b'JPEG'
    shape = list(shape)
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/x1': float_feature(x1),
            'image/object/bbox/x2': float_feature(x2),
            'image/object/bbox/x3': float_feature(x3),
            'image/object/bbox/x4': float_feature(x4),
            'image/object/bbox/y1': float_feature(y1),
            'image/object/bbox/y2': float_feature(y2),
            'image/object/bbox/y3': float_feature(y3),
            'image/object/bbox/y4': float_feature(y4),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/format': bytes_feature(image_format),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/encoded': bytes_feature(image_data)}))
            
    return example


def cvt_to_tfrecords(output_path , data_path, gt_path, records_per_file = 50000):

    fetcher = SynthTextDataFetcher(root_path = data_path, mat_path = gt_path)
    fid = 0
    image_idx = -1
    while image_idx < fetcher.num_images:
        with tf.python_io.TFRecordWriter(output_path%(fid)) as tfrecord_writer:
            record_count = 0;
            while record_count != records_per_file:
                image_idx += 1;
                if image_idx >= fetcher.num_images:
                    break;
                print "loading image %d/%d"%(image_idx + 1, fetcher.num_images)
                record = fetcher.fetch_record(image_idx);
                if record is None:
                    print '\nimage %d does not exist'%(image_idx + 1)
                    continue;
                image_path, image, txts, rect_bboxes, full_bboxes = record;
                """
                h, w = image.shape[0:-1]
                for bbox in rect_bboxes:
                    xmin, ymin, xmax, ymax = bbox;
                    xmin *= w
                    xmax *= w
                    ymin *= h
                    ymax *= h
                    util.img.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color = util.img.COLOR_WHITE)
                for wi, xys in enumerate(full_bboxes):
                    xys[0, :] = xys[0, :] * w
                    xys[1, :] = xys[1, :] * h
                    xys = xys.transpose()
                    cnts = util.img.points_to_contours(xys)
                    util.img.draw_contours(image, cnts, -1, color = util.img.COLOR_WHITE)
                    util.img.put_text(image, txts[wi],(xys[0, 1], xys[1,1]), color = util.img.COLOR_WHITE)
                util.img.imshow("%d"%(image_idx), image)
                """
                labels = len(rect_bboxes) * [1];
                difficult = len(rect_bboxes) * [0];
                truncated = len(rect_bboxes) * [0];
                image_data = tf.gfile.FastGFile(image_path, 'r').read()
                shape = image.shape
                example = _convert_to_example(image_data, labels, txts, rect_bboxes, full_bboxes, shape, difficult, truncated)
                tfrecord_writer.write(example.SerializeToString())
                record_count += 1;
                
        fid += 1;            
                    
if __name__ == "__main__":
    mat_path = util.io.get_absolute_path('~/dataset/SynthText/gt.mat')
    root_path = util.io.get_absolute_path('~/dataset/SynthText/')
    output_dir = util.io.get_absolute_path('~/dataset/SSD-tf/SynthText/')
    util.io.mkdir(output_dir);
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir,  'SynthText_%d.tfrecord'), data_path = root_path, gt_path = mat_path)
