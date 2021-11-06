import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
from utils import iou_width_height
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 416
IMGAGE_DIR = './data/VOCdevkit/VOC2012/JPEGImages'

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



ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]


class CustomVOCDataset(Dataset):
    
    def __init__(
        self, 
            csv_file='annotations.csv', 
            image_dir=IMGAGE_DIR, 
            anchors=ANCHORS, 
            image_size=IMAGE_SIZE,
            transforms=TRANSFORMS
    ):
        super().__init__()
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_file, index_col=0)
        self.df.drop(columns=['annotation_file', 'depth'])      # drops columns that are not used
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.image_size = image_size
        self.S = [self.image_size//32, self.image_size//16, self.image_size//8]
        self.unique_images = self.df['image_file'].unique()
        classes = self.df['class'].unique()
        self.label_encoder = {y:x for (x,y) in enumerate(classes)}  
        self.C = classes.shape[0]
        self.transforms = transforms
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        # returns the number of unique images in dataset
        return self.unique_images.shape[0]

    def __getitem__(self, index):

        # grabbing image from disk and get RGB array
        image_file = self.unique_images[index]
        image_path = os.path.join(self.image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if image is None:
            print("image couldn't be read")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get all the objects for the image and 
        objs = self.df.loc[self.df['image_file'] == image_file]     
        bboxes = list()
        for _, obj in objs.iterrows():
            # converts from (xmin, ymin, xmax, ymax) to (xmid, ymid, w, h)
            width = obj['xmax'] - obj['xmin']
            height = obj['ymax']- obj['ymin']
            xmid = obj['xmin'] + (width/2)
            ymid = obj['ymin'] + (height/2)

            # make the coordinates relative to whole image, range [0,1]
            xmid /= obj['width']
            ymid /= obj['height']
            width /= obj['width']
            height /= obj['height']

            label = self.label_encoder[obj['class']]

            bboxes.append([xmid, ymid, width, height, label])

        if self.transforms:
            augmentations = self.transforms(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        # for each scale, (number of anchers in that scale, scale_size, scale_size, 6)
        # 6 is for size of (p of obj, x, y, w, h, class)
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S]

        '''
        The following code segment goes through each box, calculates IoUs b/w anchors and the box,
        sorts each anchor by highest score, and then assigning an anchor to be responsible for that
        object per scale.
        '''
        for bbox in bboxes:
            iou_anchors = iou_width_height(torch.tensor(bbox[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = bbox
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (width * S, height * S)  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)

if __name__ == '__main__':
    ds = CustomVOCDataset()
    for i in range(len(ds)):
        image, targets = ds[i]
        print(image.shape)
        
    
    
