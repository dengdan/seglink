#encoding=utf-8
import numpy as np;
import tensorflow as tf
import util;
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example


def cvt_to_tfrecords(output_path , data_path, gt_path):
    image_names = util.io.ls(data_path, '.jpg')#[0:10];
    print "%d images found in %s"%(len(image_names), data_path);
    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        for idx, image_name in enumerate(image_names):
            bboxes = [];
            oriented_bboxes = []
            labels = [];
            labels_text = [];
            ignored = []
            path = util.io.join_path(data_path, image_name);
            if not util.img.is_valid_jpg(path):
                continue
            image = util.img.imread(path)
            print "\tconverting image:%s, %d/%d"%(image_name, idx, len(image_names));
            image_data = tf.gfile.FastGFile(path, 'r').read()
            #image = util.img.imread(path, rgb = True);
            shape = image.shape
            h, w = shape[0:2];
            h *= 1.0;
            w *= 1.0;
            image_name = util.str.split(image_name, '.')[0];
            gt_name = image_name + '.txt';
            gt_filepath = util.io.join_path(gt_path, gt_name);
            lines = util.io.read_lines(gt_filepath);
            for line in lines:
                spt = line.split(',')
                locs = spt[0: -1]
                xmin, ymin, bw, bh = [int(v) for v in locs]
                xmax = xmin + bw - 1
                ymax = ymin + bh - 1 
                xmin, ymin, xmax, ymax = xmin / w, ymin/ h,  xmax / w, ymax / h
                
                bboxes.append([xmin, ymin, xmax, ymax]);
                oriented_bboxes.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
                
                labels_text.append(str(spt[-1]));
                labels.append(1);
                ignored.append(0)
            example = convert_to_example(image_data, image_name, labels, ignored, labels_text, bboxes, oriented_bboxes, shape)
            tfrecord_writer.write(example.SerializeToString())
            
if __name__ == "__main__":
    root_dir = util.io.get_absolute_path('~/dataset/SCUT/SCUT_FORU_DB_Release/English2k/')
    training_data_dir = util.io.join_path(root_dir, 'word_img')
    training_gt_dir = util.io.join_path(root_dir,'word_annotation')
    output_dir = util.io.get_absolute_path('~/dataset/SSD-tf/SCUT/')
    util.io.mkdir(output_dir);
    cvt_to_tfrecords(output_path = util.io.join_path(output_dir, 'scut_train.tfrecord'), data_path = training_data_dir, gt_path = training_gt_dir)
    
