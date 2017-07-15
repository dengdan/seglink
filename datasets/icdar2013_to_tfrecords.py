#encoding=utf-8
import numpy as np;
import tensorflow as tf
import util
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example
        

def cvt_to_tfrecords(output_path , data_path, gt_path):
    image_names = util.io.ls(data_path, '.jpg')#[0:10];
    print "%d images found in %s"%(len(image_names), data_path);

    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_name in enumerate(image_names):
            oriented_bboxes = [];
            bboxes = []
            labels = [];
            labels_text = [];
            ignored = []
            path = util.io.join_path(data_path, image_name);
            print "\tconverting image: %d/%d %s"%(idx, len(image_names), image_name);
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
                gt = util.str.remove_all(line, ',')
                gt = util.str.split(gt, ' ');
                bbox = [int(gt[i]) for i in range(4)];
                xmin, ymin, xmax, ymax  = np.asarray(bbox) / [w, h, w, h];
                oriented_bboxes.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]);
                bboxes.append([xmin, ymin, xmax, ymax])z
                ignored.append(0);
                labels_text.append(line.split('"')[1]);
                labels.append(1);
            example = convert_to_example(image_data, image_name, labels, ignored, labels_text, bboxes, oriented_bboxes, shape)
            tfrecord_writer.write(example.SerializeToString())
        
if __name__ == "__main__":
    root_dir = util.io.get_absolute_path('~/dataset/ICDAR2015/Challenge2.Task123/')
    training_data_dir = util.io.join_path(root_dir, 'Challenge2_Training_Task12_Images')
    training_gt_dir = util.io.join_path(root_dir,'Challenge2_Training_Task1_GT')
    test_data_dir = util.io.join_path(root_dir,'Challenge2_Test_Task12_Images')
    test_gt_dir = util.io.join_path(root_dir,'Challenge2_Test_Task1_GT')
    
    output_dir = util.io.get_absolute_path('~/dataset/SSD-tf/ICDAR/')
    util.io.mkdir(output_dir);
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'icdar2013_train.tfrecord'), data_path = training_data_dir, gt_path = training_gt_dir)
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir,  'icdar2013_test.tfrecord'), data_path = test_data_dir, gt_path = test_gt_dir)
