#python visualize_detection_result.py \
#	--image=~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Test_Task12_Images/ \
#	--gt=~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Test_Task1_GT/ \
#	--det=~/temp/no-use/seglink_debug_icdar2013/eval/icdar2013_test/model.ckpt-48176/txt/ \
#	--output=~/temp/no-use/seglink_result

#python visualize_detection_result.py \
#	--image=~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Training_Task12_Images/ \
#	--gt=~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Training_Task1_GT/ \
#	--det=~/temp/no-use/seglink_debug_icdar2013/eval/icdar2013_train/model.ckpt-48176/txt/ \
#	--output=~/temp/no-use/seglink_result

	
python visualize_detection_result.py \
	--image=~/dataset/ICDAR2015/Challenge4/ch4_training_images/ \
	--gt=~/dataset/ICDAR2015/Challenge4/ch4_training_localization_transcription_gt/ \
	--det=~/models/seglink/seglink_icdar2015_without_ignored/eval/icdar2015_train/model.ckpt-72885/seg_link_conf_th_0.900000_0.700000/txt \
	--output=~/temp/no-use/seglink_result_512_train
	