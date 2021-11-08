import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# annotations.py
ANNOTATION_DIR = './data/VOCdevkit/VOC2012/Annotations/'
CSV_PATH = './annotations.csv'
CLASS_TARGETS = [
    'person',
    'motorbike',
    'bus',
    'car',
    'bicycle'
]

# dataset parameters
IMAGE_SIZE = 416
IMGAGE_DIR = './data/VOCdevkit/VOC2012/JPEGImages'
S = [IMAGE_SIZE//32, IMAGE_SIZE//16, IMAGE_SIZE//8]
C = len(CLASS_TARGETS)

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

TRANSFORMS = A.Compose(
    [ 
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE),
            min_width=int(IMAGE_SIZE),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)



# model hyperparameters
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 10

# paths
SAVE_PATH = './save/yolo_v3.pt'
CP_PATH = './save/yolo_v3.pt'

