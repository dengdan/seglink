from __future__ import print_function
import util
import datasets.txt2xml
import logging
def eval(gt_txt_dir, det_txt_dir, xml_path = '~/temp_nfs/results/no-use', write_path = None):
  det_xml_path = util.io.join_path(xml_path, 'det.xml')
  gt_xml_path = util.io.join_path(xml_path, 'gt.xml')
  gt_txt_dir = util.io.get_absolute_path(gt_txt_dir)
  datasets.txt2xml.txt2xml(det_txt_dir, det_xml_path);
  datasets.txt2xml.txt2xml(gt_txt_dir, gt_xml_path)
  result = util.cmd.cmd('cd %s;evalfixed %s %s'%(xml_path, det_xml_path, gt_xml_path))
  logging.info(result)
  print(result)
#  print util.cmd.cmd('cd %s;evalplots --doplot=true %s %s'%(xml_path, det_xml_path, gt_xml_path))
  if write_path is not None:
    with open(write_path, 'w') as f:
      print(result, file = f)
      print("result written to %s"%(write_path))
  return result


if __name__ == '__main__':
    gt_txt_dir = '~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Test_Task1_GT'
    det_txt_dir = '~/temp_nfs/ssd_results/model.ckpt-73483'
    evalfixed(gt_txt_dir, det_txt_dir)
    

