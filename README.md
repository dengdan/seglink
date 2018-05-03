Tips: A more recent scene text detection algorithm: [PixelLink](https://arxiv.org/abs/1801.01315), has been implemented here: https://github.com/ZJULearning/pixel_link


Contents:
1. [Introduction](https://github.com/dengdan/seglink#introduction)
2. [Installation&requirements](https://github.com/dengdan/seglink#installationrequirements)
3. [Datasets](https://github.com/dengdan/seglink#datasets)
3. [Problems](https://github.com/dengdan/seglink#problems)
5. [Models](https://github.com/dengdan/seglink#models)
4. [Test Your own images](https://github.com/dengdan/seglink#test-your-own-images)
5. [Models](https://github.com/dengdan/seglink#training-and-evaluation)
5. [Some Comments](https://github.com/dengdan/seglink#some-comments)
<hr>

# Introduction

This is a re-implementation of the SegLink text detection algorithm described in the paper [Detecting Oriented Text in Natural Images by Linking Segments, Baoguang Shi, Xiang Bai, Serge Belongie](https://arxiv.org/abs/1703.06520)



# Installation&requirements

1. tensorflow-gpu 1.1.0

2. cv2. I'm using 2.4.9.1, but some other versions less than 3 should be OK too. If not, try to switch to the version as mine.

3. download the project [pylib](https://github.com/dengdan/pylib) and add the `src` folder to your `PYTHONPATH`



If any other requirements unmet, just install them following the error msg.



# Datasets

1. [SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)

2. [ICDAR2015](http://rrc.cvc.uab.es/?ch=4&com=downloads)

Convert them into tfrecords format using the scripts in `datasets` if you wanna train your own model.



# Problems

The convergence speed of my seglink is quite slow compared with that described in the paper.  For example, the authors of SegLink paper said that a good result can be obtained by training on Synthtext for less than 10W iterations and on IC15-train for less than 1W iterations. However, using my implementation, I have to train on SynthText for about 20W iterations and another more than 10W iterations on IC15-train, to get a competitive result.

Several reasons may contribute to the slow convergency of my model:

1. Batch size. I don't have 4 12G-Titans for training, as described in the paper.  Instead, I trained my model on two 8G GeForce GTX 1080 or two Titans. 
2. Learning Rate. In the paper, 10^-3 and 10^-4 have been used. But I adopted a fixed learning rate of 10^-4.
3. Different initialization model. I used the pretrained VGG model from [SSD-caffe on coco](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6) , because I thought it better than VGG trained on ImageNet. However, it seems  that my point of view does not hold.
4.Some other differences exists maybe, I am not sure.



# Models

Two models trained on SynthText and IC15 train can be downloaded. 

1. [seglink-384](https://pan.baidu.com/s/1slqaYux). Trained using image size of  384x384, the same image size as the paper. The  Hmean is comparable to the result reported in the paper. 

![](http://fromwiz.com/share/resources/b3a92ec9-764c-470f-89a9-958c7cdeea1f/index_files/490589735.png)

The `hust_orientedText` is the result of paper.

2. [seglink-512](https://pan.baidu.com/s/1slqaYux). Trainied using image size of 512x512,  and one pointer better than 384x384. 

![](http://fromwiz.com/share/resources/0f0c6085-322f-46bc-8535-9fed33620997/index_files/1569377909.png)



They have been trained:

* on Synthtext for about 20W iterations, and on IC15-train for 10w~20W iterations. 

* learning_rate = 10e-4

* two gpus

* 384: GTX 1080, batch_size = 24; 512: Titan, batch_size = 20

**Both models perform best at `seg_conf_threshold=0.8` and `link_conf_threshold=0.5`**, well, another difference from paper, which takes 0.9 and 0.7 respectively.

# Test Your own images

Use the script `test_seglink.py`,  and a shortcut has been created in `script test.sh`:

Go to the seglink root directory and execute the command:

```

./scripts/test.sh 0 GPU_ID CKPT_PATH DATASET_DIR

```

For example:

```

./scripts/test.sh 0 ~/models/seglink/model.ckpt-217867  ~/dataset/ICDAR2015/Challenge4/ch4_training_images

```

I have only tested my models on IC15-test, but any other images can be used for test: just put your images into a directory, and config the path in the command as `DATASET_DIR`.

A bunch of txt files and a zip file is created after test. If you are using IC15-test for testing, you can upload this zip file to the [icdar evaluation server](http://rrc.cvc.uab.es/) directly.



The text files and placed in a subdir of the checkpoint directory, and contain the bounding boxes as the detection results, and can visualized using the script `visualize_detection_result.py`.

The command looks like:

```

python visualize_detection_result.py \

    --image=where your images are put

    --det=the directory of the text files output by test_seglink.py

    --output=the output directory of detection results drawn on images.

```

For example:

```

python visualize_detection_result.py \

    --image=~/dataset/ICDAR2015/Challenge4/ch4_training_images/ \

    --det=~/models/seglink/seglink_icdar2015_without_ignored/eval/icdar2015_train/model.ckpt-72885/seg_link_conf_th_0.900000_0.700000/txt \
    --output=~/temp/no-use/seglink_result_512_train

```

![](https://github.com/dengdan/seglink/blob/master/img_10_pred.jpg?raw=true)
![](https://github.com/dengdan/seglink/blob/master/img_31_pred.jpg?raw=true)

# Training and evaluation

The training processing requires data processing, i.e. converting data into tfrecords. The converting scripts are put in the `datasets` directory. The scrips:`train_seglink.py` and `eval_seglink.py` are the training and evaluation scripts respectively. Especially, I have implemented an offline evaluation function, which calculates the Recall/Precision/Hmean as the ICDAR test server, and can be used for cross validation and grid search.  However, the resulting scores may have slight differences from those of test sever, but it does not matter that much. 
Sorry for the imcomplete documentation here. Read and modify them if you want to train your own model. 



# Some Comments

Thanks should be given to the authors of the Seglink paper, i.e., Baoguang Shi1 Xiang Bai1, Serge Belongie.

[EAST](https://arxiv.org/abs/1704.03155) is another paper on text detection accepted by CVPR 2017, and its reported result is better than that of SegLink. But if they both use same VGG16, their performances are quite similar. 

Contact me if you have any problems, through github issues.

# Some Notes On Implementation Detail
How the groundtruth is calculated, in Chinese: http://fromwiz.com/share/s/34GeEW1RFx7x2iIM0z1ZXVvc2yLl5t2fTkEg2ZVhJR2n50xg

