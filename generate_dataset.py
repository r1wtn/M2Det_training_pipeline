import numpy as np
import random
from PIL import Image
import glob
import random
import argparse
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET

def create_dataset_config(data_name, class_num, ratio):
    if ratio is None:
        raise TypeError("Please indicate ratio arg")
    img_dir = 'data/VOC' + data_name + "/JPEGImages/*.jpg"
    #label_dir = data_name + "/labels/*"
    images = sorted(glob.glob(img_dir))
    random.shuffle(images)
    
    #labels = sorted(glob.glob(label_dir))

    train_len = int(len(images)*ratio)
    with open('data/VOC' + data_name + '/ImageSets/Main/' + '/trainval.txt', 'w') as f:
        for text in images[:train_len]:
            text = text.split('/')[-1]
            text = text.split('.')[0]
            f.write(text + "\n")

    with open('data/VOC' + data_name + '/ImageSets/Main/' + '/test.txt', 'w') as f:
        for text in images[train_len:]:
            text = text.split('/')[-1]
            text = text.split('.')[0]
            f.write('data/' + text + "\n")


def preprocessVoc(img_path, xml_path, offset):
    """
    shift the position of bounding boxes on both images and labels
    
    Parameters
    ----------
    img_path : path to image file
    label : path to label file
    offset : tuple(x, y) to shift
    out_name : ex) 
    
    Returns:
    img : shifted image 
    """
    print(img_path)
    img = np.asarray(Image.open(img_path).convert('RGB'))

    h, w, _ = img.shape
    
    xmlRoot = ET.parse(xml_path).getroot()
    xmlRoot.find('filename').text = img_path.split('/')[-1]

    for member in xmlRoot.findall('object'):
        bndbox = member.find('bndbox')

        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')

        # shift label boxes
        x1_ = float(xmin.text) + offset[0]
        y1_ = float(ymin.text) + offset[1]
        x2_ = float(xmax.text) + offset[0]
        y2_ = float(ymax.text) + offset[1]
        # box conditions
        cond_x1 = x1_ <= 0
        cond_y1 = y1_ <= 0
        cond_x2 = x2_ >= w
        cond_y2 = y2_ >= h

        xmin.text = str(np.round(int(x1_)))
        ymin.text = str(np.round(int(y1_)))
        xmax.text = str(np.round(int(x2_)))
        ymax.text = str(np.round(int(y2_)))

        invalid = cond_x1+cond_y1+cond_x2+cond_y2

        # judge wheter object is in bad conditions
        if member.find('occluded') is not None:
            occluded = int(member.find('occluded').text)
            invalid = cond_x1+cond_y1+cond_x2+cond_y2+occluded

        if invalid:
            xmlRoot.remove(member)

    tree = ET.ElementTree(xmlRoot)


    # shift image
    pad_x = np.zeros((h,abs(offset[0]),3))
    pad_y = np.zeros((abs(offset[1]),w,3))
    if offset[0] > 0:
        img = np.concatenate([pad_x, img], axis=1)
        img = np.delete(img, slice(w, None), axis=1)
        img = img.astype('uint8')
    else:
        img = np.concatenate([img, pad_x], axis=1)
        img = np.delete(img, slice(None, abs(offset[0])), axis=1)
        img = img.astype('uint8')
    if offset[1] > 0:
        img = np.concatenate([pad_y, img], axis=0)
        img = np.delete(img, slice(h, None), axis=0)
        img = img.astype('uint8')
    else:
        img = np.concatenate([img, pad_y], axis=0)
        img = np.delete(img, slice(None, abs(offset[1])), axis=0)
        img = img.astype('uint8')

    return img, tree


parser = argparse.ArgumentParser(description='shift image')
parser.add_argument('-i', '--input', help='path to root directory of input')
parser.add_argument('-x', '--diff_x', type=int, help='pixel to shift on x axis')
parser.add_argument('-y', '--diff_y', type=int, help='pixel to shift on y axis')
parser.add_argument('-m', '--magnification', type=int, default=5, help='magnification of images')
parser.add_argument('-c', '--class_num', type=int, default=1, help='number of classes')
parser.add_argument('-r', '--ratio', type=float, default=0.7, help='number of classes')
args = parser.parse_args()

diff_x = args.diff_x
diff_y = args.diff_y
magnification = args.magnification
class_num = args.class_num
data_name = args.input.split('/')[-1]
ratio = args.ratio

img_out = 'data/VOC' + data_name + '/JPEGImages'
xml_out = 'data/VOC' + data_name + '/Annotations'
print(img_out)
dataset_out = 'data/VOC' + data_name + '/ImageSets/Main'
images_path = sorted(glob.glob(args.input + '/VOC' + data_name + '/JPEGImages/*.jpg'))
xmls_path = sorted(glob.glob(args.input + '/VOC' + data_name + '/Annotations/*.xml'))


if os.path.exists(img_out):
    pass
else:
    os.makedirs(img_out)
    os.makedirs(xml_out)
    os.makedirs(dataset_out)

offset_list = [tuple((random.randint(-diff_x,diff_x),random.randint(-diff_y,diff_y))) for i in range(magnification)]

for i, (image_path, xml_path) in tqdm(enumerate(zip(images_path, xmls_path))):
    for j, offset in enumerate(offset_list):
        n_img, n_label = preprocessVoc(image_path, xml_path, offset)
        Image.fromarray(n_img).save(img_out+f'/{i:04d}_{j:03d}.jpg')
        n_label.write(xml_out+f'/{i:04d}_{j:03d}.xml')

create_dataset_config(data_name, class_num, ratio)