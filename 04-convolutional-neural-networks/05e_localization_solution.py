import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder

# Set directories
IMAGE_DIR = "pets/images"
ANNOTATION_DIR = "pets/annotations/xmls"

IMG_SIZE = 224  # Input size for MobileNetV2

# Function to preprocess images (pad, resize, normalize)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    # Make square by padding with black pixels
    pad_size = max(h, w)
    pad_image = np.zeros((pad_size, pad_size, 3), dtype=np.uint8)
    pad_image[:h, :w, :] = image

    # Resize to network input size
    resized_image = cv2.resize(pad_image, (IMG_SIZE, IMG_SIZE))

    return resized_image, (h, w, pad_size)

# Function to extract bounding box and labels from XML
def parse_annotation(xml_path, original_dims, padded_size):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = root.findall("object")
    if not objects:
        return None  # Ignore images without bounding boxes

    bboxes = []
    labels = []
    original_h, original_w, _ = original_dims

    for obj in objects:
        label = obj.find("name").text
        bbox = obj.find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Scale bounding box coordinates
        scale_x = IMG_SIZE / padded_size
        scale_y = IMG_SIZE / padded_size

        xmin = int(xmin * scale_x)
        ymin = int(ymin * scale_y)
        xmax = int(xmax * scale_x)
        ymax = int(ymax * scale_y)

        bboxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return bboxes, labels

# Load dataset
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
X_images = []
Y_bboxes = []
Y_labels = []

for image_file in image_files:
    image_path = os.path.join(IMAGE_DIR, image_file)
    xml_path = os.path.join(ANNOTATION_DIR, image_file.replace(".jpg", ".xml"))

    if not os.path.exists(xml_path):
        continue  # Ignore images without annotation files

    # Preprocess image
    processed_image, dims = preprocess_image(image_path)
    
    # Parse annotation
    annotation = parse_annotation(xml_path, dims, max(dims))
    
    if annotation is None:
        continue  # Ignore images without bounding boxes

    bboxes, labels = annotation

    X_images.append(processed_image)
    Y_bboxes.append(bboxes[0])  # Assume one bounding box per image
    Y_labels.append(labels[0])  # Assume one class per image

# Convert to NumPy arrays
X_images = np.array(X_images) / 255.0  # Normalize pixel values
Y_bboxes = np.array(Y_bboxes)  # Bounding boxes

# Encode labels as numbers
label_encoder = LabelEncoder()
Y_labels_encoded = label_encoder.fit_transform(Y_labels)

# Load MobileNetV2 as the base model (without classification head)
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze layers

# Define input layer
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Feature extraction
x = base_model(inputs, training=False)
x = Flatten()(x)

# Classification Head (Softmax for multiple breeds)
num_classes = len(set(Y_labels_encoded))
class_output = Dense(num_classes, activation="softmax", name="class_output")(x)

# Bounding Box Regression Head (4 values: xmin, ymin, xmax, ymax)
bbox_output = Dense(4, activation="sigmoid", name="bbox_output")(x)

# Create model with two outputs
model = Model(inputs, outputs=[class_output, bbox_output])

# Compile model with two losses
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={"class_output": "sparse_categorical_crossentropy", "bbox_output": "mse"},
    metrics={"class_output": "accuracy", "bbox_output": "mse"}
)

# Show model architecture
model.summary()

# Train the model
history = model.fit(
    X_images, 
    {"class_output": Y_labels_encoded, "bbox_output": Y_bboxes}, 
    epochs=10, 
    batch_size=32,
    validation_split=0.2
)

# Evaluate model
loss, class_loss, bbox_loss, class_acc, bbox_mse = model.evaluate(
    X_images, {"class_output": Y_labels_encoded, "bbox_output": Y_bboxes}
)
print(f"Classification Accuracy: {class_acc:.2f}")
print(f"Bounding Box MSE: {bbox_mse:.4f}")

# Predict on a new image
sample_image = np.expand_dims(X_images[0], axis=0)
pred_class, pred_bbox = model.predict(sample_image)

# Decode label
predicted_label = label_encoder.inverse_transform([np.argmax(pred_class)])

print(f"Predicted Label: {predicted_label[0]}")
print(f"Predicted Bounding Box: {pred_bbox[0]}")
