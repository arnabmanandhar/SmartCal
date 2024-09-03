# Import required libraries
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger, Callback
import csv

# Root directory of the project
ROOT_DIR = "D:\Nepali_Food_Detection"

# Import Mask RCNN library
sys.path.append(ROOT_DIR)  # Add ROOT_DIR to system path to access Mask R-CNN locally

# Path to the pre-trained COCO weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Create the logs directory if it doesn't exist
if not os.path.exists(DEFAULT_LOGS_DIR):
    os.makedirs(DEFAULT_LOGS_DIR)

# Custom configuration for training on the custom Nepali food dataset
class CustomConfig(Config):
    """Configuration class for training the custom dataset.
    This class defines parameters for training Mask R-CNN on a custom dataset.
    """
    # Name for the configuration
    NAME = "object"

    # Number of images processed per GPU
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + 4 food classes (egg, rice, lentils, spinach)

    # Batch size
    BATCH_SIZE = 4

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 58

    # Minimum confidence level for detections
    DETECTION_MIN_CONFIDENCE = 0.9

# Dataset class for the custom food dataset
class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset.
        Args:
            dataset_dir: Root directory of the dataset
            subset: Subset of the dataset to load ("train" or "val")
        """
        # Add classes (food categories)
        self.add_class("object", 1, "friedegg")
        self.add_class("object", 2, "rice")
        self.add_class("object", 3, "lentils")
        self.add_class("object", 4, "spinach")

        # Determine if we're loading the training or validation dataset
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations from JSON file
        annotations1 = json.load(open(os.path.join(dataset_dir, 'via_region_data.json')))
        annotations = list(annotations1.values())  # Convert dict values to list
        annotations = [a for a in annotations if a['regions']]  # Filter out images without regions

        # Add images to the dataset
        for a in annotations:
            image_id = "{}_{}".format(subset, a['filename'])
            image_path = os.path.join(dataset_dir, a['filename'])
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            # Extract polygon information for each region
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['name'] for s in a['regions']]
            name_dict = {"friedegg": 1, "rice": 2, "lentils": 3, "spinach": 4}
            num_ids = [name_dict[a] for a in objects]

            # Add the image to the dataset
            self.add_image(
                "object", image_id=image_id, path=image_path,
                width=width, height=height, polygons=polygons, num_ids=num_ids
            )

        self.prepare()  # Prepare the dataset (required before training)

    def load_image(self, image_id):
        """Load the specified image.
        Args:
            image_id: The image ID to load
        Returns:
            The image as a NumPy array
        """
        image = cv2.imread(self.image_info[image_id]['path'])
        return image

    def load_mask(self, image_id):
        """Generate instance masks for the image.
        Args:
            image_id: The image ID to load masks for
        Returns:
            masks: A binary array with one mask per instance
            class_ids: A 1D array of class IDs for the instances
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        num_ids = image_info['num_ids']
        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])], dtype=np.uint8)

        # Create mask for each polygon
        for i, p in enumerate(image_info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            rr = np.clip(rr, 0, image_info["height"] - 1)
            cc = np.clip(cc, 0, image_info["width"] - 1)
            mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            return super(self.__class__, self).image_reference(image_id)

# Function to train the model
def train(model, initial_epoch=0):
    # Load training and validation datasets
    dataset_train = CustomDataset()
    dataset_train.load_custom("D:\\Nepali_Food_Detection\\dataset", "train")

    dataset_val = CustomDataset()
    dataset_val.load_custom("D:\\Nepali_Food_Detection\\dataset", "val")

    # Load the last saved model checkpoint (if continuing training)
    if initial_epoch > 0:
        model_path = model.find_last()
        model.load_weights(model_path, by_name=True)

    # Log training details to CSV
    csv_logger = CSVLogger(os.path.join(DEFAULT_LOGS_DIR, 'history.csv'), append=True)

    # Train the network's heads
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=initial_epoch + 100,  # Update total number of epochs
                layers='heads',
                custom_callbacks=[csv_logger])

# Initialize the configuration
config = CustomConfig()

# Create the Mask R-CNN model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)

# Load pre-trained COCO weights
weights_path = COCO_WEIGHTS_PATH

# Download weights if not already available
if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

# Load the COCO weights and exclude specific layers that will be retrained
model.load_weights(weights_path, by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Train the model starting from epoch 51
train(model, initial_epoch=0)
