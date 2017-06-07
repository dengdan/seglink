#encoding=utf-8
import matplotlib as mpl
mpl.use('Agg')
import util;
import numpy as np;
class ICDAR2013Data(object):
    def __init__(self, root_dir = '~/dataset/ICDAR2015/Challenge2.Task123/',
        training_data_dir = 'Challenge2_Training_Task12_Images',
        training_gt_dir = 'Challenge2_Training_Task1_GT',
        split = 'test',
        test_data_dir = 'Challenge2_Test_Task12_Images',
        test_gt_dir = 'Challenge2_Test_Task1_GT'):
        self.split = split
        self.image_idx = -1;
        if split == 'test':
            self.gt_path = util.io.join_path(root_dir, test_gt_dir)
            self.img_path = util.io.join_path(root_dir, test_data_dir)
        else:
            self.gt_path = util.io.join_path(root_dir, training_gt_dir)
            self.img_path = util.io.join_path(root_dir, training_data_dir)
            
            
        images, bboxes, image_names = self.get_images_and_gt(gt_path = self.gt_path, data_path = self.img_path);
        self.num_images = len(images)
        self.data = [[img, bboxes, image_name] for(img, bboxes, image_name) in zip(images, bboxes, image_names)]
    
    def reset(self):
        self.image_idx = -1;
    def get_images_and_gt(self, data_path, gt_path):
        image_names = util.io.ls(data_path, '.jpg')#[0:10];
        print "%d images found in %s"%(len(image_names), data_path);
        images = [];
        bboxes = [];
        aspect_ratios = []
        heights_rel = []
        heights_abs = []
        for idx, image_name in enumerate(image_names):
            path = util.io.join_path(data_path, image_name);
            print "\treading image: %d/%d"%(idx, len(image_names));
            image = util.img.imread(path, rgb = True);
#            image = image / 255.0
            images.append(image);
            image_name = util.str.split(image_name, '.')[0];
            gt_name = 'gt_' + image_name + '.txt';
            gt_filepath = util.io.join_path(gt_path, gt_name);
            lines = util.io.read_lines(gt_filepath);
            bbox_gt = []
            h, w = image.shape[0:2];
            h *= 1.0;
            w *= 1.0;
            for line in lines:
                gt = util.str.remove_all(line, ',');
                gt = util.str.split(gt, ' ');
                box =[int(gt[i]) for i in range(4)];
                x1, y1, x2, y2  = box;
                box = [y1 / h, x1 / w, y2 / h,  x2 / w];
                bbox_gt.append(box);
                aspect_ratios.append(((x2 - x1)*1.0 / w )/((y2 - y1)*1.0 / h))
                heights_rel.append((y2 - y1) * 1.0 / h)
                heights_abs.append((y2 - y1) * 1.0 / h * 512)
                #print y2 - y1, h, (y2 - y1) * 1.0 / h
            bbox_gt = np.asarray(bbox_gt)
            bboxes.append(bbox_gt);
            image_names[idx] = image_name
#            if idx >= 1:
#                break;
        
#        util.io.dump('~/temp/no-use/as_%s.pkl'%(self.split), aspect_ratios)
        self.aspect_ratios = aspect_ratios;
        self.heights_rel = heights_rel
        self.heights_abs = heights_abs
        return images, bboxes, image_names;
        
    def vis_data(self):
        for image, bboxes in self.data:
            for bbox in bboxes:
                x1, y1, x2, y2  = bbox;
                util.img.rectangle(image, (x1, y1), (x2, y2), color = util.img.COLOR_WHITE, border_width = -1);
            
            util.img.imshow("", image);
    
    def get_data(self):
        self.image_idx += 1;
        if self.image_idx >= len(self.data):
            util.np.shuffle(self.data);
            self.image_idx = 0;
        image, bboxes, name = self.data[self.image_idx];
        labels = [1] * len(bboxes);
        labels = np.reshape(labels, [-1, 1]);
        return image, bboxes, labels, name;


def vis_data():
    import numpy as np
    import tensorflow as tf
    import time
    import util
    util.mod.add_to_path('..')
    from preprocessing.preprocessing_factory  import get_preprocessing
    data_provider = ICDAR2013Data();
    
    util.proc.set_proc_name('proc-test');
    
    fn = get_preprocessing(True);
    with tf.Graph().as_default():
        sess = tf.Session();
        sess.as_default();
        out_shape = [150, 150]
        images = tf.placeholder("float32", name = 'images', shape = [None, None, 3])
        bboxes = tf.placeholder("float32", name = 'bboxes', shape = [None, 4])
        labels = tf.placeholder('int32', name = 'labels', shape = [None, 1])
        
        sampled_image, sampled_labels, sampled_bboxes = fn(images, labels, bboxes, out_shape);
        step = 0;
        data = []
        while step < len(data_provider.num_images):
            step += 1;
            start = time.time();
            image_data, bbox_data, label_data = data_provider.get_data();
            feed_dict = {images: image_data, labels: label_data, bboxes: bbox_data}
            I, L, B = sess.run([sampled_image, sampled_labels, sampled_bboxes], feed_dict = feed_dict)
            I = np.asarray(I, dtype = np.uint8);
            B *= 150;
            I_box = I.copy()
            for bbox in B:
                y1, x1, y2, x2 = bbox
                util.img.rectangle(I_box, (x1, y1), (x2, y2), color = util.img.COLOR_WHITE)
            end = time.time();
            util.plt.show_images(images = [I, I_box], save = True, show = False, path = '~/temp_nfs/no-use/%d.jpg'%(step))
#            data.append([image_data, I_, L]);
#            util.io.dump('test.pkl', data);

def aspect_ratio_cal(split,k):
    """
    statistics of aspect ratios in icdar training and test data.
    split: train or test
    k: the N.o. centers of aspect ratios
    """
    print split
    data_provider = ICDAR2013Data(split=split);
    aspect_ratios = data_provider.aspect_ratios
    labels, clusters, centers = util.ml.kmeans(aspect_ratios, k);
    for i in xrange(len(centers)):
        print 'Aspect ratio: %f, n: %d, ratio: %f'%(centers[i], len(clusters[i]),len(clusters[i]) * 1./len(aspect_ratios))

    heights = data_provider.heights_rel
    labels, clusters, centers = util.ml.kmeans(heights, k);
    for i in xrange(len(centers)):
        print 'Relative Height: %f, n: %d, ratio: %f'%(centers[i], len(clusters[i]),len(clusters[i]) * 1./len(aspect_ratios))
    hist, bins = np.histogram(heights, range=[0, 0.2])
    for i in xrange(len(hist)):
        print 'Relative Height between %f and %f: n = %d, ratio = %f'%(bins[i], bins[i+1], hist[i], hist[i]  * 1.0/ len(heights))

    heights = data_provider.heights_abs
    labels, clusters, centers = util.ml.kmeans(heights, k);
    for i in xrange(len(centers)):
        print 'Absolute Height: %f, n: %d, ratio: %f'%(centers[i], len(clusters[i]),len(clusters[i]) * 1./len(aspect_ratios))
    hist, bins = np.histogram(heights, range=[0, 52])
    for i in xrange(len(hist)):
        print 'Absolute Height between %f and %f: n = %d, ratio = %f'%(bins[i], bins[i+1], hist[i], hist[i]  * 1.0/ len(heights))
        
if __name__ == "__main__":
    #aspect_ratio_cal('test', 3);
    aspect_ratio_cal('train', 3);
    
