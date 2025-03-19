import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import glob
import time

OUTPUT = ".\\output" # No need to change
#SAMPLES = ".\\samples" # normal
SAMPLES = ".\\samples2" # testing (smaller dataset)
EPOCHS = 15

class FaceTrackingModelTrainer:
    def __init__(self):
        self.data_dir = OUTPUT
        self.samples_dir = SAMPLES
        self.img_size = (64, 64)
        self.batch_size = 32
        self.model = None
        
        
    def build_model(self):
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))

        # conv 1
        x = Conv2D(32, (3, 3), activation="relu")(inputs)
        x = MaxPooling2D((2, 2))(x)

        # conv2
        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)

        # conv 3
        x = Conv2D(128, (3, 3), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)

        # connect conv
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation="relu")(x)

        outputs = Dense(1, activation="sigmoid")(x) 

        # create with the input & conv
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        self.model = model
        return model
    
    def process_samples(self):
        print(f"Processing face images from {self.samples_dir}...")
        
        # object detection model
        fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        # get all jpgs
        files = glob.glob(os.path.join(self.samples_dir, "*.jpg")) 
        
        print(f"Found {len(files)} image files in {self.samples_dir}")
        
        count = 0
        t = time.time()
        for img_path in files:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"couldnt read {img_path}")
                    continue
                
                # easier for model when only a color depth of 1
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # look for faces 
                faces = fc.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) == 0:
                    face_img = cv2.resize(img, self.img_size)
                    filename = os.path.join(self.data_dir, "faces", f"face_{count}.jpg")
                    cv2.imwrite(filename, face_img)
                    count += 1
                else:
                    for (x, y, w, h) in faces: # X, Y, width and length (clarification)
                        face_img = img[y:y+h, x:x+w]
                        
                        face_img = cv2.resize(face_img, self.img_size)
                        
                        filename = os.path.join(self.data_dir, "faces", f"face_{count}.jpg")
                        cv2.imwrite(filename, face_img)
                        count += 1
                et = time.time() - t
                imgas = count//et
                print(f"processed {count} faces from {len(files)} images | {imgas} img/s | ET: {(len(files) - count)//imgas}s", end="\r")
                
            except Exception as e:
                print(f"error processing {img_path}: {str(e)}")
        
        print(f"\nfinished processing {count} faces from {len(files)} images.")
        return count
    
    def prepare_data(self):
        print("Preparing data for training...")
        
        face_dir = os.path.join(self.data_dir, 'faces')
        face_images = []
        
        for filename in os.listdir(face_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(face_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                    img = img / 255.0 
                    face_images.append(img)
        
        if not face_images:
            raise ValueError("No face images found in the faces directory.")
        
        face_images = np.array(face_images)
        
        labels = np.ones((len(face_images), 1))
        
        X_train, X_val, y_train, y_val = train_test_split(
            face_images, labels, test_size=0.2, random_state=42
        )
        
        return X_train, X_val, y_train, y_val
    
    def train_model(self, epochs=20):
        if self.model is None:
            self.build_model()
        
        X_train, X_val, y_train, y_val = self.prepare_data()
        
        # generate new data from the data we already have to test
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        
        print("Training model...")
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=len(X_train) // self.batch_size if len(X_train) >= self.batch_size else 1,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        # graph 1
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        # graph 2
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, "training_history.png"))
        plt.show()
    
    def save_model(self):
        filename="face_tracking_model"
        mp = os.path.join(self.data_dir, filename)
        mp_keras = mp + ".keras"
        mp_h5 = mp + ".h5"
        
        tf.keras.models.save_model(self.model, mp_keras)
        print(f"Model saved to {mp_keras}")
        
        tf.keras.models.save_model(self.model, mp_h5)
        print(f"Model also saved to {mp_h5}")

        c = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = c.convert()
        tflite_mp = mp + ".tflite"
        
        with open(tflite_mp, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {tflite_mp}")
        
        return mp_keras

def main():
    trainer = FaceTrackingModelTrainer()
    
    trainer.process_samples()
    
    trainer.build_model()
    trainer.train_model(epochs=EPOCHS)
    mp = trainer.save_model()
    
    print(f"Training done")

main()