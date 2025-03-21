import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import glob
import time
import random

OUTPUT = ".\\output"  # No need to change
SAMPLES = ".\\samples3"
EPOCHS = 30  # Increased from 3 to 15 for better convergence
NEGATIVE_SAMPLES_RATIO = 1.0  # Equal number of negative samples
LEARNING_RATE = 0.0001 # basically the speed the model learns at (DO NOT TOUCH if you dont know what you are doing)

class FaceTrackingModelTrainer:
    def __init__(self):
        self.data_dir = OUTPUT
        self.samples_dir = SAMPLES
        self.img_size = (64, 64)
        self.batch_size = 32
        self.model = None
        
        os.makedirs(os.path.join(self.data_dir, "faces"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "non_faces"), exist_ok=True)
        
    #def build_model(self):
    #    inputs = Input(shape=(self.img_size[0], self.img_size[1], 3))

    #    # first conv
    #    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    #    x = BatchNormalization()(x)
    #    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    #    x = BatchNormalization()(x)
    #    x = MaxPooling2D((2, 2))(x)
    #    x = Dropout(0.2)(x)

    #    # second conv
    #    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    #    x = BatchNormalization()(x)
    #    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    #    x = BatchNormalization()(x)
    #    x = MaxPooling2D((2, 2))(x)
    #    x = Dropout(0.3)(x)

    #    # third conv
    #    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    #    x = BatchNormalization()(x)
    #    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    #    x = BatchNormalization()(x)
    #    x = MaxPooling2D((2, 2))(x)
    #    x = Dropout(0.4)(x)

    #    # flatten and dense layers
    #    x = Flatten()(x)
    #    x = Dense(256, activation="relu")(x)
    #    x = BatchNormalization()(x)
    #    x = Dropout(0.5)(x)
    #    x = Dense(128, activation="relu")(x)
    #    x = BatchNormalization()(x)
    #    x = Dropout(0.5)(x)

    #    outputs = Dense(1, activation="sigmoid")(x)

    #    model = Model(inputs=inputs, outputs=outputs)

    #    model.compile(optimizer=Adam(LEARNING_RATE),
    #                 loss="binary_crossentropy",
    #                 metrics=["accuracy"])

    #    self.model = model
    #    model.summary()
    #    return model
    
    def process_samples(self):
        print(f"Processing face images from {self.samples_dir}...")

        fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        files = glob.glob(os.path.join(self.samples_dir, "*.jpg")) 
        if not files:
            files = glob.glob(os.path.join(self.samples_dir, "*.png"))
        if not files:
            files = glob.glob(os.path.join(self.samples_dir, "*.jpeg"))

        print(f"Found {len(files)} image files in {self.samples_dir}")

        face_count = 0
        non_face_count = 0
        t_start = time.time()
        total_files = len(files)

        for idx, img_path in enumerate(files):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Couldn't read {img_path}")
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                if len(faces) == 0:
                    continue 
                else:
                    for (x, y, w, h) in faces:
                        margin = int(w * 0.1) 
                        x_start = max(0, x - margin)
                        y_start = max(0, y - margin)
                        x_end = min(img.shape[1], x + w + margin)
                        y_end = min(img.shape[0], y + h + margin)

                        face_img = img[y_start:y_end, x_start:x_end]

                        if face_img.size == 0:
                            continue

                        try:
                            face_img = cv2.resize(face_img, self.img_size)
                            filename = os.path.join(self.data_dir, "faces", f"face_{face_count}.jpg")
                            cv2.imwrite(filename, face_img)
                            face_count += 1

                            if random.random() < NEGATIVE_SAMPLES_RATIO:
                                for _ in range(1): 
                                    max_attempts = 5
                                    for attempt in range(max_attempts):
                                        nx = random.randint(0, max(0, img.shape[1] - self.img_size[0]))
                                        ny = random.randint(0, max(0, img.shape[0] - self.img_size[1]))
                                        nw, nh = self.img_size

                                        overlap = False
                                        for (fx, fy, fw, fh) in faces:
                                            ix1 = max(nx, fx)
                                            iy1 = max(ny, fy)
                                            ix2 = min(nx + nw, fx + fw)
                                            iy2 = min(ny + nh, fy + fh)
                                            if ix2 > ix1 and iy2 > iy1:
                                                overlap = True
                                                break

                                        if not overlap:
                                            non_face_img = img[ny:ny+nh, nx:nx+nw]
                                            if non_face_img.shape[:2] == self.img_size:
                                                filename = os.path.join(self.data_dir, "non_faces", f"non_face_{non_face_count}.jpg")
                                                cv2.imwrite(filename, non_face_img)
                                                non_face_count += 1
                                            break
                        except Exception as e:
                            print(f"Error processing face: {str(e)}")

                elapsed = time.time() - t_start
                processed = face_count + non_face_count
                img_per_sec = processed / elapsed if elapsed > 0 else 0
                time_per_img = elapsed / max(1, idx + 1)
                est_total_time = total_files * time_per_img
                est_remaining = est_total_time - elapsed
                print(f"[{idx+1}/{total_files}] Processed {face_count} faces and {non_face_count} non-faces | {img_per_sec:.1f} img/s | Est. remaining: {est_remaining:.1f}s", end="\r")

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

        total_elapsed = time.time() - t_start
        print(f"\nFinished processing {face_count} faces and {non_face_count} non-faces in {total_elapsed / 60:.2f} minutes ({total_elapsed:.1f} seconds).")
        return face_count, non_face_count
    
    def prepare_data(self):
        print("Preparing data for training...")
        
        face_dir = os.path.join(self.data_dir, 'faces')
        face_images = []
        face_count = 0
        
        for filename in os.listdir(face_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')): # i gave options incase you dont want to use the data source i suggested
                img_path = os.path.join(face_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size) 
                    img = img / 255.0 
                    face_images.append(img)
                    face_count += 1
        
        non_face_dir = os.path.join(self.data_dir, 'non_faces')
        non_face_images = []
        non_face_count = 0
        
        if os.path.exists(non_face_dir):
            for filename in os.listdir(non_face_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(non_face_dir, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.img_size)
                        img = img / 255.0
                        non_face_images.append(img)
                        non_face_count += 1
        
        print(f"Loaded {face_count} face images and {non_face_count} non-face images")
        
        X = []
        y = []
        
        if face_images:
            X.extend(face_images)
            y.extend([1] * len(face_images))
        
        if non_face_images:
            X.extend(non_face_images)
            y.extend([0] * len(non_face_images))
        
        X = np.array(X)
        y = np.array(y)
        
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # starting training

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training data: {X_train.shape[0]} samples")
        print(f"Validation data: {X_val.shape[0]} samples")
        
        return X_train, X_val, y_train, y_val
    
    #def train_model(self, epochs=EPOCHS):
    #    if self.model is None:
    #        self.build_model()
    #    
    #    X_train, X_val, y_train, y_val = self.prepare_data()
    #    
    #    datagen = ImageDataGenerator(
    #        rotation_range=30,
    #        width_shift_range=0.2,
    #        height_shift_range=0.2,
    #        shear_range=0.2,
    #        zoom_range=0.2,
    #        horizontal_flip=True,
    #        brightness_range=[0.5, 1.5], 
    #        fill_mode="nearest"
    #    )
    #    
    #    callbacks = [
    #        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    #        
    #        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    #        
    #        ModelCheckpoint(
    #            filepath=os.path.join(self.data_dir, 'best_model.h5'),
    #            monitor='val_accuracy',
    #            save_best_only=True,
    #            verbose=1
    #        )
    #    ]
    #    
    #    print("Training model...")
    #    history = self.model.fit(
    #        datagen.flow(X_train, y_train, batch_size=self.batch_size),
    #        steps_per_epoch=len(X_train) // self.batch_size if len(X_train) >= self.batch_size else 1,
    #        epochs=epochs,
    #        validation_data=(X_val, y_val),
    #        verbose=1,
    #        callbacks=callbacks
    #    )
    #    
    #    self.plot_training_history(history)
    #    
    #    best_model_path = os.path.join(self.data_dir, 'best_model.h5')
    #    if os.path.exists(best_model_path):
    #        self.model = tf.keras.models.load_model(best_model_path)
    #        print("Loaded best model from checkpoint")
    #    
    #    val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=1)
    #    print(f"Validation accuracy: {val_accuracy:.4f}")
    #    
    #    y_pred = self.model.predict(X_val)
    #    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    #    
    #    true_positives = np.sum((y_val == 1) & (y_pred_binary == 1))
    #    false_positives = np.sum((y_val == 0) & (y_pred_binary == 1))
    #    true_negatives = np.sum((y_val == 0) & (y_pred_binary == 0))
    #    false_negatives = np.sum((y_val == 1) & (y_pred_binary == 0))
    #    
    #    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    #    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    #    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    #    
    #    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
    #    print(f"True Positives: {true_positives}, False Positives: {false_positives}")
    #    print(f"True Negatives: {true_negatives}, False Negatives: {false_negatives}")
    #    
    #    return history
    
    #def plot_training_history(self, history):
    #    plt.figure(figsize=(12, 8))
    #    
    #    plt.subplot(2, 1, 1)
    #    plt.plot(history.history["loss"], label="Training Loss")
    #    plt.plot(history.history["val_loss"], label="Validation Loss")
    #    plt.title("Training and Validation Loss")
    #    plt.xlabel("Epoch")
    #    plt.ylabel("Loss")
    #    plt.legend()
    #    
    #    # Plot training & validation accuracy
    #    plt.subplot(2, 1, 2)
    #    plt.plot(history.history["accuracy"], label="Training Accuracy")
    #    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    #    plt.title("Training and Validation Accuracy")
    #    plt.xlabel("Epoch")
    #    plt.ylabel("Accuracy")
    #    plt.legend()
    #    
    #    plt.tight_layout()
    #    plt.savefig(os.path.join(self.data_dir, "training_history.png"))
    #    plt.show()
    
    def validate_on_sample_images(self, num_samples=10):
        face_dir = os.path.join(self.data_dir, 'faces')
        non_face_dir = os.path.join(self.data_dir, 'non_faces')
        
        face_files = [os.path.join(face_dir, f) for f in os.listdir(face_dir) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
        non_face_files = [os.path.join(non_face_dir, f) for f in os.listdir(non_face_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if face_files:
            face_samples = random.sample(face_files, min(num_samples, len(face_files)))
        else:
            face_samples = []
            
        if non_face_files:
            non_face_samples = random.sample(non_face_files, min(num_samples, len(non_face_files)))
        else:
            non_face_samples = []
        
        # starting visualisation
        n_samples = len(face_samples) + len(non_face_samples)
        if n_samples == 0:
            print("No samples to validate on")
            return
            
        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 3, 3))
        if n_samples == 1:
            axes = [axes]
        # this is where the cool testing with the results of the visualiation
        for i, img_path in enumerate(face_samples + non_face_samples):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, self.img_size)
            img_normalized = img_resized / 255.0
            
            prediction = self.model.predict(np.expand_dims(img_normalized, axis=0))[0][0]
            is_face = prediction >= 0.5
            
            expected = img_path in face_samples
            
            axes[i].imshow(img_resized)
            if is_face == expected:
                result = "Correct"
                color = "green"
            else:
                result = "Wrong"
                color = "red"

            if is_face:
                predicted_label = "Face"
            else:
                predicted_label = "Non-face"

            if expected:
                true_label = "Face"
            else:
                true_label = "Non-face"

            title = f"{result}\nPred: {predicted_label} ({prediction:.2f})\nTrue: {true_label}"

            axes[i].set_title(title, color=color)
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, "validation_samples.png"))
        plt.show()
    
    def save_model(self):
        base_path = os.path.join(self.data_dir, "face_tracking_model")
        
        # h5
        h5_path = base_path + ".h5"
        self.model.save(h5_path)
        print(f"Model saved to {h5_path}")
        
        # keras
        keras_path = base_path + ".keras"
        self.model.save(keras_path)
        print(f"Model saved to {keras_path}")
        
        return h5_path
    
    def train_model_improved(self): 
        if self.model is None:
            self.build_simpler_model()

        X_train, X_val, y_train, y_val = self.prepare_data()

        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        print(f"Training data: {pos_count} faces, {neg_count} non-faces")

        weight_for_0 = (1 / neg_count) * (len(y_train) / 2.0) if neg_count > 0 else 1.0
        weight_for_1 = (1 / pos_count) * (len(y_train) / 2.0) if pos_count > 0 else 1.0
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f"Class weights: {class_weight}")

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(
                filepath=os.path.join(self.data_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True
            )
        ]

        print("Training improved model...")
        start_time = time.time()
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=callbacks,
            class_weight=class_weight
        )
        total_time = time.time() - start_time
        print(f"Training completed in {total_time / 60:.2f} minutes ({total_time:.1f} seconds)")

        return history

    
    
    
    def build_simpler_model(self):
        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', 
                        input_shape=(self.img_size[0], self.img_size[1], 3)))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))  
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        model.summary()
        self.model = model
        return model




def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    trainer = FaceTrackingModelTrainer()
    face_count, non_face_count = trainer.process_samples()
    if face_count == 0:
        print("No face samples were processed. Please check your sample images.")
        return

    trainer.build_simpler_model()
    trainer.train_model_improved()
        
    trainer.validate_on_sample_images()
        
    model_path = trainer.save_model()
        
    print(f"Training complete! Model saved to {model_path}")





if __name__ == "__main__":
    main()