import sys
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image
import dearpygui.dearpygui as dpg
from tensorflow.keras.models import load_model

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Load the face detection cascade using resource_path
face_cascade = cv2.CascadeClassifier(resource_path("haarcascade_frontalface_default.xml"))

# Optional debug check
print("Cascade loaded:", not face_cascade.empty())

# Load model and face detector
IMG_SIZE = 224
label_map = ['surprised', 'happy', 'angry', 'disgust', 'fear', 'sad', 'neutral']
suspicious_labels = ['sad', 'angry', 'fear', 'surprised']

model = load_model(resource_path("best_model3_acc_h5final.h5"), compile=False)

cap = cv2.VideoCapture(0)
os.makedirs("captured_faces", exist_ok=True)

# GUI image buffer
texture_width = 650
texture_height = 500
frame_data = np.zeros((texture_height, texture_width, 4), dtype=np.float32)  # RGBA

# Dear PyGui setup
dpg.create_context()
dpg.create_viewport(title='Emotion Detection App', width=1000, height=600)
dpg.setup_dearpygui()

with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(texture_width, texture_height, frame_data.flatten(), tag="live_texture")

with dpg.window(tag="main_window", label="Emotion Detection App", width=1000, height=600):
    with dpg.group(horizontal=True):
        dpg.add_image("live_texture", width=texture_width, height=texture_height)
        with dpg.child_window(tag="right_panel", width=300, height=500):
            dpg.add_text("Suspicious Faces:")
    dpg.add_button(label="Stop", callback=lambda: stop_app())

# Store thumbnails to avoid duplication
shown_thumbs = set()
app_running = True

def capture_suspicious_face(face, label):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"captured_faces/{label}_{timestamp}.jpg"
    cv2.imwrite(filename, face)
    return filename

def show_thumbnail(image_path):
    if image_path in shown_thumbs:
        return
    shown_thumbs.add(image_path)

    thumb = Image.open(image_path).resize((80, 80)).convert("RGBA")
    thumb_data = np.array(thumb).astype(np.float32) / 255.0
    tag = f"thumb_texture_{len(shown_thumbs)}"

    with dpg.texture_registry(show=False):
        dpg.add_static_texture(80, 80, thumb_data.flatten(), tag=tag)

    dpg.add_image(tag, width=80, height=80, parent="right_panel")

def frame_callback():
    if not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        input_tensor = np.reshape(resized, (1, IMG_SIZE, IMG_SIZE, 3))

        try:
            _, pred = model.predict(input_tensor, verbose=0)
        except:
            pred = model.predict(input_tensor, verbose=0)

        idx = np.argmax(pred[0])
        label = label_map[idx]
        conf = np.max(pred[0]) * 100

        if conf < 65:
            continue

        color = (0, 255, 0)
        text = f"{label} ({conf:.1f}%)"

        if label in suspicious_labels:
            color = (0, 0, 255)
            text += " - Suspicious"
            thumb_path = capture_suspicious_face(face_roi, label)
            show_thumbnail(thumb_path)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    rgba_frame = cv2.cvtColor(cv2.resize(frame, (texture_width, texture_height)), cv2.COLOR_BGR2RGBA)
    rgba_norm = rgba_frame.astype(np.float32) / 255.0
    dpg.set_value("live_texture", rgba_norm.flatten())

def stop_app():
    global app_running
    app_running = False
    cap.release()

dpg.set_primary_window("main_window", True)
dpg.show_viewport()

# ✅ NEW MAIN LOOP (replaces set_render_callback)
while dpg.is_dearpygui_running() and app_running:
    frame_callback()
    dpg.render_dearpygui_frame()

dpg.destroy_context()
