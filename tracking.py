import cv2
import numpy as np
import tensorflow as tf
import time
import os

MODEL_PATH = "output/face_tracking_model"

class FaceDetector:
    def __init__(self, model_path, confidence_threshold):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.img_size = (64, 64)
        
        self.model = self.load_model()
        
        # Load Haar cascade as a fast first-pass detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def load_model(self):
        """Load the trained face detection model"""
        try:
            if os.path.exists(f"{self.model_path}.keras"):
                print(f"Loading Keras model from {self.model_path}.keras")
                model = tf.keras.models.load_model(f"{self.model_path}.keras")
                return model
            else:
                raise FileNotFoundError(f"Model file not found at {self.model_path}.keras")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)  # Exit if model is not loaded
    
    def preprocess_image(self, img):
        """Preprocess image for model input"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.img_size)
        img_normalized = img_resized / 255.0
        return np.expand_dims(img_normalized, axis=0)
    
    def predict_with_model(self, img):
        """Make prediction with TensorFlow model"""
        preprocessed = self.preprocess_image(img)
        prediction = self.model.predict(preprocessed, verbose=0)
        return prediction[0][0]
    
    def detect_faces(self, frame):
        # Resize frame for faster processing
        scale = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Use Haar cascade as a fast first-pass filter
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detected_faces = []
        
        # For each potential face region found by Haar cascade
        for (x, y, w, h) in faces:
            # Convert coordinates back to original scale
            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            
            # Add some padding around the detected face
            padding = int(w * 0.1)
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(frame.shape[1] - x_pad, w + 2*padding)
            h_pad = min(frame.shape[0] - y_pad, h + 2*padding)
            
            # Extract the region and verify with our model
            roi = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            if roi.size == 0:
                continue
                
            try:
                confidence = self.predict_with_model(roi)
                if confidence >= self.confidence_threshold:
                    detected_faces.append((x_pad, y_pad, w_pad, h_pad, float(confidence)))
            except Exception as e:
                print(f"Prediction error: {e}")
        
        return detected_faces
    
    def run_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        # Set lower resolution for faster processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        start_time = time.time()
        fps = 0
        process_every_n_frames = 1  # Process every frame now that we're faster
        frame_index = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            # Only process every nth frame
            if frame_index % process_every_n_frames == 0:
                detected_faces = self.detect_faces(frame)
                
                for (x, y, w, h, confidence) in detected_faces:
                    # Color coding based on confidence
                    if confidence > 0.98:
                        color = (0, 255, 0)  # Green for high confidence
                    elif confidence > 0.95:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 0, 255)  # Red for lower confidence
                    
                    # Only blur high confidence detections
                    if confidence > 0.98:
                        face_region = display_frame[y:y+h, x:x+w]
                        if face_region.size > 0:  # Ensure the region is valid
                            kernel_size = max(1, min(w, h) // 10) * 2 + 1  # Must be odd number
                            face_region_blurred = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 30)
                            display_frame[y:y+h, x:x+w] = face_region_blurred

                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    label = f"{confidence:.2f}"
                    cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                frame_count += 1
            
            frame_index += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Threshold: {self.confidence_threshold:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Face Detection (Optimized)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            # Quit on 'q'
            if key == ord("q"):
                break
            # Increase threshold on '+'
            elif key == ord("+") or key == ord("="):
                self.confidence_threshold = min(0.99, self.confidence_threshold + 0.01)
                print(f"Confidence threshold increased to {self.confidence_threshold:.2f}")
            # Decrease threshold on '-'
            elif key == ord("-") or key == ord("_"):
                self.confidence_threshold = max(0.5, self.confidence_threshold - 0.01)
                print(f"Confidence threshold decreased to {self.confidence_threshold:.2f}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = FaceDetector(model_path=MODEL_PATH, confidence_threshold=0.97)
    detector.run_camera()

if __name__ == "__main__":
    main()