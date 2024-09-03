import os
import numpy as np
import pandas as pd
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from sklearn.metrics import f1_score

class CustomConfig(Config):
    # Your configuration settings go here
    """Configuration for training on the custom dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + egg, rice, lentils, spinach

    BATCH_SIZE = 4

    IMAGE_META_SIZE = 23
  
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 78

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

# Root directory of the project
ROOT_DIR = "D:\Aarohi"


class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "egg")
        self.add_class("object", 2, "rice")
        self.add_class("object", 3, "lentils")
        self.add_class("object", 4, "spinach")
        # self.add_class("object", 5, "pasta")
        # self.add_class("object", 6, "dosa")
        # self.add_class("object", 7, "momo")
        # self.add_class("object", 8, "roti")
        # self.add_class("object", 9, "chai")
        # self.add_class("object", 10, "jalebi")


        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations1 = json.load(open(os.path.join(dataset_dir, 'via_region_data.json')))
        annotations = list(annotations1.values())
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            image_id = "{}_{}".format(subset, a['filename'])
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            

            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes']['name'] for s in a['regions']]
            name_dict = {"egg": 1, "rice": 2, "lentils": 3, "spinach":4}
            num_ids = [name_dict[a] for a in objects]

            self.add_image(
                "object",
                image_id=image_id,
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids
            )

# Directory containing validation dataset
VAL_DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "val")

# CSV file to store the results
RESULTS_CSV_FILE = os.path.join(ROOT_DIR, "results.csv")

# Load the custom config
config = CustomConfig()

# Load the validation dataset
dataset_val = CustomDataset()
dataset_val.load_custom(VAL_DATASET_DIR, "val")
dataset_val.prepare()

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Epoch', 'F1_Score'])

# Loop through all saved model weights files
for epoch in range(1, 51):  # Assuming you have 50 epochs
    # Construct the path to the model weights file for the current epoch
    model_weights_path = os.path.join(ROOT_DIR, "logs", f"mask_rcnn_object_{epoch:04d}.h5")

    # Create and load the model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=ROOT_DIR)
    model.load_weights(model_weights_path, by_name=True)

    # Predict on the validation dataset
    y_true = []
    y_pred = []

    for image_id in dataset_val.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
            dataset_val, config, image_id, use_mini_mask=False)

        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        outputs = model.predict([molded_images, image_meta], verbose=0)
        r = model.detect([image], verbose=0)[0]

        # Extract predicted class IDs
        pred_class_ids = r['class_ids']

        # Convert true class IDs to match prediction format
        true_class_ids = [dataset_val.map_source_class_id("object.{}".format(cls_id), "object") for cls_id in gt_class_id]

        y_true.extend(true_class_ids)
        y_pred.extend(pred_class_ids)

    # Calculate F1 score for the current epoch
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Append the results to the DataFrame
    results_df = results_df.append({'Epoch': epoch, 'F1_Score': f1}, ignore_index=True)

    print(f"Epoch {epoch}: F1 Score: {f1}")

# Save the results to a CSV file
results_df.to_csv(RESULTS_CSV_FILE, index=False)
