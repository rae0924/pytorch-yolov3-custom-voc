import xml.etree.ElementTree as ET
import pandas as pd
import glob
import os

class_targets = [
    'person',
    'motorbike',
    'bus',
    'car',
    'bicycle'
]

annotations_dir = './data/VOCdevkit/VOC2012/Annotations/'
csv_path = './annotations.csv'

annotation_file_list, image_file_list, class_list = [], [], []
width_list, height_list, depth_list = [], [], []
xmin_list, ymin_list, xmax_list, ymax_list = [], [], [], []
for xmlfilepath in glob.glob(annotations_dir+'*.xml'):
    xmlfile = ET.parse(xmlfilepath)
    root = xmlfile.getroot()
    annotation_file = os.path.basename(xmlfilepath)
    image_file = root.find('filename').text
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    depth = float(size.find('depth').text)
    # width, height, depth = [float(d.text) for d in root.find('size')] doesn't work because of tags are not alway in this order

    for obj in [child for child in root if child.tag == 'object']:
        class_name = obj.find('name').text
        if class_name in class_targets:
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            #xmin, ymin, xmax, ymax = [float(d.text) for d in obj.find('bndbox')] doesn't work because of tags are not alway in this order
            annotation_file_list.append(annotation_file)
            image_file_list.append(image_file)
            class_list.append(class_name)
            width_list.append(width)
            height_list.append(height)
            depth_list.append(depth)
            xmin_list.append(xmin)
            ymin_list.append(ymin)
            xmax_list.append(xmax)
            ymax_list.append(ymax)

df = pd.DataFrame(
    {
        'annotation_file': annotation_file_list,
        'image_file': image_file_list,
        'class': class_list,
        'width': width_list,
        'height': height_list,
        'depth': depth_list,
        'xmin': xmin_list,
        'ymin': ymin_list,
        'xmax': xmax_list,
        'ymax': ymax_list
    }
)           

df.to_csv(csv_path)