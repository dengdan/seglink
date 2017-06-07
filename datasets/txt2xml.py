#encoding=utf-8

import xml.dom.minidom
import util
import re
pattern = re.compile('\d+')
def get_image_name(txt_name, postfix = 'jpg'):
    match = pattern.search(txt_name)
    return match.group() + "." + postfix


def txt2xml(txt_dir, output_path, use_existed = False):
    """  convert all text files into one xml file
    :param txt_dir: the input path of text files
    :param image_dir: the test images corresponding to the text files
    :param output_path: the name of save path of xml file
    :use_existed: whether to use an existed output file if it existed.
    :return: none

    example:
    """
    if use_existed and util.io.exists(output_path):
        return output_path
    
    output_path = util.io.get_absolute_path(output_path);
    util.io.make_parent_dir(output_path);
    
    txt_files = util.io.ls(txt_dir, '.txt');

    # 创建xml
    impl = xml.dom.minidom.getDOMImplementation()
    tagset = impl.createDocument(None, 'tagset', None)
    tagset_val = tagset.documentElement

    
    for txt_name in txt_files:
        # 获取图像名
        imageName = get_image_name(txt_name);

        # 写入image
        imageNode = tagset.createElement('image')
        tagset_val.appendChild(imageNode)
        
        # 写入imageName
        imageNameNode = tagset.createElement('imageName')
        imageNameNode_val = tagset.createTextNode(imageName)
        imageNameNode.appendChild(imageNameNode_val)
        imageNode.appendChild(imageNameNode)

        # 读取txt文件,将里面的位置信息写入xml
        tagNode = tagset.createElement('taggedRectangles')
        imageNode.appendChild(tagNode)
        
        gt_path = util.io.join_path(txt_dir, txt_name);
        lines = util.io.read_lines(gt_path)
        rect_list = [];
        for line in lines:
            line = util.str.remove_all(line, ',');
            data = line.split()
            x1, y1, x2, y2 = [int(d) for d in data[:4]]
            width = x2 - x1;
            height = y2 - y1
            rect_list.append((x1, y1, width, height))

        rects = list(set(rect_list))
        assert len(rects) == len(rect_list), "len(rects) != len(rect_list)"
        for rect in rects:
            x, y, width, height = rect
            locNode = tagset.createElement('taggedRectangle')
            locNode.setAttribute('x', str(x))
            locNode.setAttribute('y', str(y))
            locNode.setAttribute('width', str(width))
            locNode.setAttribute('height', str(height))
            locNode.setAttribute('offset', str(0))
            tagNode.appendChild(locNode)

    with open(output_path, 'w') as f:
        tagset.writexml(f, addindent='  ', newl='\n')


if __name__ == "__main__":
    txt_dir = '~/dataset/ICDAR2015/Challenge2.Task123/Challenge2_Training_Task1_GT'  # text文件路径
    txt_dir = '~/temp_nfs/ssd_results/model.ckpt-73483/'
    output_path = '~/temp_nfs/results/test.xml'  # xml保存路径和文件名
    txt2xml(txt_dir, output_path)


