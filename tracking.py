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
        # this will try to use a GPU if u have it
        
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
        self.model = self.load_model()
        
        self.last_detections = []
        
    def load_model(self):
        try:
            if os.path.exists(f"{self.model_path}.keras"):
                print(f"loading model from {self.model_path}.keras")
                model = tf.keras.models.load_model(f"{self.model_path}.keras")
                return model
            else:
                raise FileNotFoundError(f"model not found at {self.model_path}.keras")
        except Exception as e:
            print(f"error loading model: {e}")
            return None
    
    def preprocess_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img_rgb, self.img_size)

        img_normalized = img_resized / 255.0

        return np.expand_dims(img_normalized, axis=0)
    
    def predict_with_model(self, img):
        preprocessed = self.preprocess_image(img)
        prediction = self.model.predict(preprocessed, verbose=0)
        return prediction[0][0]
    
    def batch_predict(self, images):
        if not images:
            return []
        
        processed_images = []
        for img in images:
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, self.img_size)
                img_normalized = img_resized / 255.0
                processed_images.append(img_normalized)
            except Exception:
                processed_images.append(np.zeros(self.img_size + (3,)))
        
        batch = np.array(processed_images)
        
        predictions = self.model.predict(batch, verbose=0)
        
        return predictions.flatten()
    
    def detect_faces(self, frame, full_scan=False):
        frame_height, frame_width = frame.shape[:2]
        
        if self.last_detections and not full_scan:
            rois = []
            roi_positions = []
            
            for x, y, w, h, _ in self.last_detections:
                pad_w, pad_h = int(w * 0.5), int(h * 0.5)
                start_x = max(0, x - pad_w)
                start_y = max(0, y - pad_h)
                end_x = min(frame_width, x + w + pad_w)
                end_y = min(frame_height, y + h + pad_h)
                
                roi = frame[start_y:end_y, start_x:end_x]
                if roi.size > 0:
                    rois.append(roi)
                    roi_positions.append((start_x, start_y, end_x - start_x, end_y - start_y))
            
            if rois:
                confidences = self.batch_predict(rois)
                detected_faces = []
                
                for i, conf in enumerate(confidences):
                    if conf >= self.confidence_threshold:
                        x, y, w, h = roi_positions[i]
                        detected_faces.append((x, y, w, h, float(conf)))
                
                if detected_faces:
                    self.last_detections = detected_faces
                    return detected_faces
        
        detected_faces = []
        rois = []
        roi_positions = []
        
        base_size = min(frame_width, frame_height) // 4
        window_sizes = [
            (base_size, base_size),
            (int(base_size * 1.5), int(base_size * 1.5)),
            (int(base_size * 2), int(base_size * 2))
        ]
        
        step_ratio = 0.5
        
        for window_size in window_sizes:
            step_x = int(window_size[0] * step_ratio)
            step_y = int(window_size[1] * step_ratio)
            
            for y in range(0, frame_height - window_size[1] + 1, step_y):
                for x in range(0, frame_width - window_size[0] + 1, step_x):
                    roi = frame[y:y + window_size[1], x:x + window_size[0]]
                    if roi.size > 0:
                        rois.append(roi)
                        roi_positions.append((x, y, window_size[0], window_size[1]))
                    
                    if len(rois) >= 50:
                        break
                if len(rois) >= 50:
                    break
            if len(rois) >= 50:
                break
        
        if rois:
            confidences = self.batch_predict(rois)
            
            for i, conf in enumerate(confidences):
                if conf >= self.confidence_threshold:
                    detected_faces.append(roi_positions[i] + (float(conf),))
        
        if detected_faces:
            boxes = np.array([face[:4] for face in detected_faces])
            confidences = np.array([face[4] for face in detected_faces])
            
            boxes_for_nms = []
            for (x, y, w, h) in boxes:
                boxes_for_nms.append([x, y, x + w, y + h])
            boxes_for_nms = np.array(boxes_for_nms)
            
            indices = tf.image.non_max_suppression(
                boxes_for_nms, 
                confidences, 
                max_output_size=10, 
                iou_threshold=0.5
            ).numpy()
            
            final_faces = []
            for i in indices:
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                final_faces.append((int(x), int(y), int(w), int(h), float(confidence)))
            
            self.last_detections = final_faces
            return final_faces
        
        self.last_detections = []
        return []
    
    def run_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera this will not work if you dont have one ")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        process_every_n_frames = 5
        frame_index = 0
        
        # check the ENTIRE screen every so often
        full_scan_interval = 30
        full_scan_counter = 0
        
        # decrease resolution to better performance
        downscale_factor = 0.5
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            if frame_index % process_every_n_frames == 0:
                # more performance
                process_frame = cv2.resize(frame, (0, 0), fx=downscale_factor, fy=downscale_factor)
                
        
                do_full_scan = (full_scan_counter % full_scan_interval == 0)
                
                # the detecting faces
                detected_faces_small = self.detect_faces(process_frame, full_scan=do_full_scan)
                
                # increase resolution after decreasing it
                detected_faces = []
                for x, y, w, h, conf in detected_faces_small:
                    scaled_x = int(x / downscale_factor)
                    scaled_y = int(y / downscale_factor)
                    scaled_w = int(w / downscale_factor)
                    scaled_h = int(h / downscale_factor)
                    detected_faces.append((scaled_x, scaled_y, scaled_w, scaled_h, conf))
                
                full_scan_counter += 1
                
                if detected_faces:
                    self.last_detections = detected_faces
            
            faces_to_draw = self.last_detections
            # color coding
            for (x, y, w, h, confidence) in faces_to_draw:
                if confidence > 0.98:
                    color = (0, 255, 0) 
                elif confidence > 0.95:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)
                
                if confidence > 0.75:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(display_frame.shape[1], x + w), min(display_frame.shape[0], y + h)
                    
                    if x2 > x1 and y2 > y1: 
                        face_region = display_frame[y1:y2, x1:x2]
                        if face_region.size > 0: 
                            kernel_size = max(1, min(x2-x1, y2-y1) // 10) * 2 + 1 
                            # blur effect
                            face_region_blurred = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 30)
                            display_frame[y1:y2, x1:x2] = face_region_blurred

                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                label = f"{confidence:.2f}"
                cv2.putText(display_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            frame_index += 1
            if frame_index % process_every_n_frames == 0:
                frame_count += 1
            
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Threshold: {self.confidence_threshold:.2f} (+ to increase | - to decrease)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Face Detection (Optimized)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("+") or key == ord("="):
                self.confidence_threshold = min(0.99, self.confidence_threshold + 0.01)
                print(f"new thresholgd {self.confidence_threshold:.2f}")
            elif key == ord("-") or key == ord("_"):
                self.confidence_threshold = max(0.5, self.confidence_threshold - 0.01)
                print(f"new threshold {self.confidence_threshold:.2f}")
            elif key == ord("f"):
                print("forcing full refresh")
                full_scan_counter = 0
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = FaceDetector(model_path=MODEL_PATH, confidence_threshold=0.97)
    detector.run_camera()

if __name__ == "__main__":
    main()