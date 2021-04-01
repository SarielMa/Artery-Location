from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import json


def getEuclidean(a, b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
def my_evaluate(model,valid_path="data/samples2/validation" ):
    parser = argparse.ArgumentParser()
    #parser.add_argument("--image_folder", type=str, default="data/samples2/test", help="path to dataset")
    #parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    #parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_50.pth", help="path to weights file")
    #parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    #parser.add_argument("--conf_thres", type=float, default=0.25, help="object confidence threshold")
    #parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    #parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    #parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    #parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    #parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    #print(opt)
    class_path = "data/custom/classes.names"
    conf_thres=0.25
    nms_thres=0.3
    img_size = 416
    label_path = "data/sample2/labels"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)

    # Set up model
    """
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    """

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(valid_path, transform= transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])),
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )

    classes = load_classes(class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        #print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    
    distances = []
    discovers = 0
    shouldbe=0
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        img = np.array(Image.open(path))
        
        base =path.split("validation")[0]
        left = base+"valid_labels/"
        right = path.split("validation/")[1].split(".")[0]+".json"
        label = left+right
        lumen = []
        with open(label, "rb") as f:
            conts = json.load(f)
            lumen = conts["lumen"]
        centroids = [[np.mean(np.array(l)[:,0]), np.mean(np.array(l)[:,1])] for l in lumen]
        shouldbe +=(len(centroids))
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, cls_conf, cls_pred in detections:
                discovers+=1
                #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                
                # get the smallest distance
                cx = ((x1+x2)/2).cpu().item()
                cy = ((y1+y2)/2).cpu().item()
                
                if len(centroids)>0:
                    m=1440
                    index = -1
                    for i,c in enumerate(centroids):
                        distance = getEuclidean(c, [cx,cy])
                        if distance<m:
                            m = distance
                            index = i
                    distances.append(m)
                    centroids.remove(centroids[index])
                #draw boxes
    mean = 0
    if len(distances)>0:
        mean = np.mean(np.array(distances))
    else:
        mean = 720
    print("mean distance is :", mean)
    print("detections: ", discovers)
    print("objects: ",shouldbe)
    return mean
