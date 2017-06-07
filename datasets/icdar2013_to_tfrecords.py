#encoding=utf-8
import numpy as np;
import tensorflow as tf
import util;
from dataset_utils import int64_feature, float_feature, bytes_feature
        
def _convert_to_example(image_data, labels, labels_text, bboxes, shape, difficult, truncated):
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
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
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
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/format': bytes_feature(image_format),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/encoded': bytes_feature(image_data)}))
            
    return example


def cvt_to_tfrecords(output_path , data_path, gt_path):
    image_names = util.io.ls(data_path, '.jpg')#[0:10];
    print "%d images found in %s"%(len(image_names), data_path);

    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_name in enumerate(image_names):
            bboxes = [];
            labels = [];
            labels_text = [];
            difficult = []
            truncated = []
            path = util.io.join_path(data_path, image_name);
            print "\treading image: %d/%d"%(idx, len(image_names));
            image_data = tf.gfile.FastGFile(path, 'r').read()
            
            image = util.img.imread(path, rgb = True);
            shape = image.shape
            h, w = shape[0:2];
            h *= 1.0;
            w *= 1.0;
            image_name = util.str.split(image_name, '.')[0];
            gt_name = 'gt_' + image_name + '.txt';
            gt_filepath = util.io.join_path(gt_path, gt_name);
            lines = util.io.read_lines(gt_filepath);
            for line in lines:
                gt = util.str.remove_all(line, ',');
                gt = util.str.split(gt, ' ');
                box =[int(gt[i]) for i in range(4)];
                x1, y1, x2, y2  = box;
                box = [y1 / h, x1 / w, y2 / h,  x2 / w];
                bboxes.append(box);
                labels_text.append(gt[-1]);
                labels.append(1);
                difficult.append(0)
                truncated.append(0)
            example = _convert_to_example(image_data, labels, labels_text, bboxes, shape, difficult, truncated)
            tfrecord_writer.write(example.SerializeToString())
        
if __name__ == "__main__":
    root_dir = util.io.get_absolute_path('~/dataset/ICDAR2015/Challenge2.Task123/')
    training_data_dir = util.io.join_path(root_dir, 'Challenge2_Training_Task12_Images')
    training_gt_dir = util.io.join_path(root_dir,'Challenge2_Training_Task1_GT')
    test_data_dir = util.io.join_path(root_dir,'Challenge2_Test_Task12_Images')
    test_gt_dir = util.io.join_path(root_dir,'Challenge2_Test_Task1_GT')
    
    output_dir = util.io.get_absolute_path('~/dataset/SSD-tf/ICDAR/')
    util.io.mkdir(output_dir);
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'icdar2013_train.tfrecords'), data_path = training_data_dir, gt_path = training_gt_dir)
#    cvt_to_tfrecords(output_path = util.io.join_path(output_dir,  'icdar2013_test.tfrecords'), data_path = test_data_dir, gt_path = test_gt_dir)
