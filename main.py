import cv2
import numpy as np
import time
from picamera2 import Picamera2, Preview
import tflite_runtime.interpreter as tflite
import sqlite3
from datetime import datetime

MODEL_PATH = "plant_disease_model.tflite"
LABEL_PATH = "plant_disease_labels.txt"
DATABASE_PATH = "disease_log.db"

IMG_WIDTH = 224
IMG_HEIGHT = 224

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
LINE_TYPE = 2

def init_database():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            prediction_label TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at {DATABASE_PATH}")

def log_detection(label, confidence):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO detections (timestamp, prediction_label, confidence) VALUES (?, ?, ?)",
                   (timestamp, label, confidence))
    conn.commit()
    conn.close()

def get_recent_history(limit=3):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, prediction_label, confidence FROM detections ORDER BY id DESC LIMIT ?", (limit,))
    history = cursor.fetchall()
    conn.close()
    return reversed(history)

def load_model_and_labels(model_path, label_path):
    try:
        with open(label_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model and labels loaded successfully.")
        return interpreter, labels
    except Exception as e:
        print(f"Error loading model or labels: {e}")
        exit()

def preprocess_image(image, input_details):
    resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    input_data = np.expand_dims(resized_image, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5
    if input_details[0]['dtype'] == np.uint8:
        input_data = np.uint8(input_data * 127.5 + 127.5)
    return input_data

def main():
    init_database()
    interpreter, labels = load_model_and_labels(MODEL_PATH, LABEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (800, 600)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    print("Camera initialized. Starting real-time detection...")

    while True:
        im = picam2.capture_array()
        image_for_display = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
        input_data = preprocess_image(im, input_details)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        top_k_index = results.argmax()
        
        prediction_label = labels[top_k_index]
        confidence = float(results[top_k_index])
        
        log_detection(prediction_label, confidence)
        
        status_text = f'Current: {prediction_label.replace("___", " ")} ({confidence:.2f})'
        display_color = (0, 0, 255) if 'healthy' not in prediction_label.lower() else (0, 255, 0)
        cv2.putText(image_for_display, status_text, (10, 30), FONT, FONT_SCALE, display_color, LINE_TYPE)

        cv2.putText(image_for_display, "Progression History:", (10, 70), FONT, FONT_SCALE, (255, 255, 255), LINE_TYPE)
        history = get_recent_history(limit=3)
        y_pos = 95
        for record in history:
            timestamp, label, conf = record
            short_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%m/%d %H:%M")
            history_text = f'- {short_time}: {label.split("___")[-1]} ({conf:.2f})'
            cv2.putText(image_for_display, history_text, (15, y_pos), FONT, FONT_SCALE - 0.1, (255, 255, 0), LINE_TYPE)
            y_pos += 25
        
        cv2.imshow('Plant Disease Detection - Progression Monitor', image_for_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()
    print("Application stopped.")

if __name__ == '__main__':
    main()
