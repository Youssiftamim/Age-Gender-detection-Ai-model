import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

model_path = "D:/calc/3.11/age_gender_classification_model_v4_cpu.keras"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")

model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

AGE_LABELS = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"]
GENDER_LABELS = ["Male", "Female"] 

def preprocess_frame(frame, img_size=(48, 48)):
    frame_resized = cv2.resize(frame, img_size)
    frame_normalized = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def detect_age_gender(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            x = max(0, x)
            y = max(0, y)
            w = min(w, iw - x)
            h = min(h, ih - y)

            face = image[y:y+h, x:x+w]
            if face.size == 0:
                continue

            processed_face = preprocess_frame(face, img_size=(48, 48))
            predictions = model.predict(processed_face, verbose=0)
            age_pred = predictions[0]  
            gender_pred = predictions[1]  

            age_idx = np.argmax(age_pred)
            age_label = AGE_LABELS[age_idx]
            age_confidence = age_pred[0][age_idx] * 100

            gender_idx = np.argmax(gender_pred)
            gender_label = GENDER_LABELS[gender_idx]  
            gender_confidence = gender_pred[0][gender_idx] * 100

            
            print(f"Gender Confidence - Male: {gender_pred[0][0] * 100:.2f}%, Female: {gender_pred[0][1] * 100:.2f}%")

            
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            label = f"Age: {age_label} ({age_confidence:.1f}%)"
            cv2.putText(image, label, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            label = f"Gender: {gender_label} ({gender_confidence:.1f}%)"
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image

def webcam_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = "Age and Gender Detection (Webcam)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        
        frame = detect_age_gender(frame)

        
        cv2.imshow(window_name, frame)

        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")

def image_upload_mode():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

  
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Could not load the image.")
        return

    
    processed_image = detect_age_gender(image)

    
    window_name = "Age and Gender Detection (Image)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_gui():
    root = tk.Tk()
    root.title("Age and Gender Detection")
    root.geometry("400x300")
    root.configure(bg="#f0f0f0")

    
    title_label = tk.Label(
        root,
        text="Age and Gender Detection",
        font=("Helvetica", 20, "bold"),
        bg="#f0f0f0",
        fg="#333333"
    )
    title_label.pack(pady=20)

    
    webcam_button = tk.Button(
        root,
        text="Use Webcam",
        command=webcam_mode,
        font=("Helvetica", 14),
        bg="#4CAF50",
        fg="white",
        activebackground="#45a049",
        width=20,
        height=2
    )
    webcam_button.pack(pady=10)


    upload_button = tk.Button(
        root,
        text="Upload Image",
        command=image_upload_mode,
        font=("Helvetica", 14),
        bg="#2196F3",
        fg="white",
        activebackground="#1e88e5",
        width=20,
        height=2
    )
    upload_button.pack(pady=10)

    exit_button = tk.Button(
        root,
        text="Exit",
        command=root.quit,
        font=("Helvetica", 14),
        bg="#f44336",
        fg="white",
        activebackground="#e53935",
        width=20,
        height=2
    )
    exit_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
    print("it works <3.")
