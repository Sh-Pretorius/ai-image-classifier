import cv2
import tensorflow as tf
import sys

# Get image from user
if len(sys.argv) < 2:
    print("Usage: python imageRecog.py <image_path>")
    exit()

image_path = sys.argv[1]

image = cv2.imread(image_path) # Load an image

# Check if image is loaded
if image is None:
    print("Image could not load")
    exit()
else:
    print("Image loaded successfully")

# Preprocessing Images
processed_image = cv2.resize(image, (224, 224)) # Resize original color image
processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB (important for TensorFlow)

# Load necessary models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Load object detector
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True) # Load pre-trained model

# Preprocess image for model
img_array = tf.keras.preprocessing.image.img_to_array(processed_image) # Image converts to array
img_array = tf.expand_dims(img_array, 0) 
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# Make Prediction
predictions = model.predict(img_array)
decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)

# Saving results
cv2.imwrite('processed_image.jpg', processed_image) # saving processed image
with open('results.txt', 'w') as f:
    for pred in decoded[0]:
        f.write(f"{pred[1]}: {pred[2]*100:.2f}%\n") # write predictions to text file

top_label = decoded[0][0][1]
top_confidence = float(decoded[0][0][2]) * 100

# Print mostlikely result in terminal
print(f"Top Prediction: {top_label} ({top_confidence:.2f}%)")