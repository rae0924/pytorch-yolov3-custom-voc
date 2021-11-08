# pytorch-yolov3-custom-voc

My take on implementing YOLOv3 in PyTorch. Most of it is my own code, but some functions were inspired or entirely borrowed from this
[repository](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLOv3).

## Dataset
The dataset used for this project is a custom dataset that is a subset of the PASCAL VOC 2012 dataset. After downloading the online dataset
I picked out classes that were of particular interest to me, such as the following 5:
* 'person'
* 'motorbike'
* 'bus'
* 'car'
* 'bicycle'

as defined in config.py. Annotations.py takes in those classes and filters out the annotations. The script will pull together all relevant 
bounding boxes and their annotations and dump it into a CSV file that can be understood easily by the wrapper in dataset.py. 

## Quick Setup
* Clone this github repository
* Download the training/validation dataset from [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#voc2012vs2011) into an appropriate directory and then unzip it.
* Edit config.py to make sure to provide the chosen parameters. 
* Run annotations.py.
* Now run train.py for training the model.


## Requirements
Python3 packages:
* torch
* opencv-python
* albumentations
* pandas

all other packages such as numpy should be installed automatically under these packages' sub-requirements.
