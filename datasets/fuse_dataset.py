import util

ic15_train_path = '~/dataset/SSD-tf/ICDAR/icdar2015_train.tfrecord'
ic13_train_path = '~/dataset/SSD-tf/ICDAR/icdar2013_train.tfrecord'
ic13_test_path = '~/dataset/SSD-tf/ICDAR/icdar2013_test.tfrecord'
scut_train_path = '~/dataset/SSD-tf/SCUT/scut_train.tfrecord'

synthtext_dir = '~/dataset/SSD-tf/SynthText'

cmd = 'cd %s;ln -s %s link_%d_%s'

to_be_fused = [ic15_train_path, ic13_train_path, ic13_test_path]
repeats = [50, 10, 10]

for repeat, data_path in zip(repeats, to_be_fused):
    data_path = util.io.get_absolute_path(data_path)
    data_name = util.io.get_filename(data_path)
    
    for idx in range(repeat):
        ln_cmd = cmd%(synthtext_dir, data_path, idx, data_name)
        print ln_cmd
        print util.cmd.cmd(ln_cmd)
    