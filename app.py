import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from psutil import virtual_memory

# =====================================================
# MEMORY OPTIMIZATION
# =====================================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            tf.config.set_logical_device_configuration(
                device, [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
            )
    else:
        total_ram = virtual_memory().total / (1024 ** 3)
        print(f"[INFO] System RAM: {total_ram:.1f} GB — limiting TensorFlow threads.")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception as e:
    print(f"[WARN] Could not set memory limits: {e}")

# =====================================================
# FLASK SETUP
# =====================================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =====================================================
# MODEL & CLASSES
# =====================================================
MODEL_PATH = 'plant_disease_model.keras'
DATASET_TRAIN_PATH = 'train'

if os.path.exists(DATASET_TRAIN_PATH):
    CLASS_NAMES = sorted(os.listdir(DATASET_TRAIN_PATH))
else:
    CLASS_NAMES = [
        'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
        'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two-spotted_spider_mite', 'Tomato__Target_Spot',
        'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
        'Tomato_healthy'
    ]

# =====================================================
# 🌿 CROP CONDITIONS DATA
# =====================================================
CROP_INFO = {
    "Pepper": {
        "temperature": "20–30°C",
        "humidity": "50–70%",
        "condition": "Warm climate, good sunlight",
        "soil": "Loamy soil, well-drained"
    },
    "Potato": {
        "temperature": "15–20°C",
        "humidity": "60–80%",
        "condition": "Cool climate, loose soil",
        "soil": "Sandy loam soil"
    },
    "Tomato": {
        "temperature": "20–27°C",
        "humidity": "50–70%",
        "condition": "Warm climate, fertile soil",
        "soil": "Loamy soil rich in organic matter"
    },
    "Apple": {
        "temperature": "18–24°C",
        "humidity": "60–70%",
        "condition": "Cool climate, good sunlight",
        "soil": "Well-drained loamy soil"
    },
    "Corn": {
        "temperature": "18–30°C",
        "humidity": "50–60%",
        "condition": "Warm climate, full sunlight",
        "soil": "Sandy loam soil"
    },
    "Grape": {
        "temperature": "15–30°C",
        "humidity": "Low to Moderate",
        "condition": "Dry climate, sunlight",
        "soil": "Well-drained sandy soil"
    }
}

# =====================================================
# DUMMY MODEL
# =====================================================
def create_dummy_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =====================================================
# LOAD MODEL
# =====================================================
def load_model_safe():
    try:
        if not os.path.exists(MODEL_PATH):
            return create_dummy_model()
        return load_model(MODEL_PATH)
    except:
        return create_dummy_model()

model = load_model_safe()

CONFIDENCE_THRESHOLD = 0.80

# =====================================================
# IMAGE PREPROCESSING
# =====================================================
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# =====================================================
# LEAF DETECTION
# =====================================================
def is_leaf_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (224 * 224)

    return green_ratio > 0.1

# =====================================================
# ROUTES
# =====================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    try:
        # STEP 1: LEAF CHECK
        if not is_leaf_image(filepath):
            return jsonify({
                'predicted_class': "Invalid Image (Not a Leaf)",
                'confidence': "0%"
            })

        # STEP 2: PREPROCESS
        processed = preprocess_image(filepath)

        # STEP 3: PREDICTION
        predictions = model.predict(processed)
        pred_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        # STEP 4: CONFIDENCE CHECK
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'predicted_class': "Low Confidence - Invalid Image",
                'confidence': f"{confidence * 100:.2f}%"
            })

        # STEP 5: VALID RESULT → SHOW CONDITIONS
        predicted_label = CLASS_NAMES[pred_index]

        crop_name = predicted_label.split("_")[0]

        crop_details = CROP_INFO.get(crop_name, {
            "temperature": "N/A",
            "humidity": "N/A",
            "condition": "No data available"
        })
        
        return jsonify({
            'predicted_class': predicted_label.replace("_", " "),
            'confidence': f"{confidence * 100:.2f}%",
            'temperature': crop_details["temperature"],
            'humidity': crop_details["humidity"],
            'condition': crop_details["condition"],
            'soil': crop_details["soil"]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# =====================================================
# MAIN
# =====================================================
if __name__ == '__main__':
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True)