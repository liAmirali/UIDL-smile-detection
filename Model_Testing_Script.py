import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf

from mtcnn import MTCNN

class FaceDetection:
    # Custom layer to convert RGB to grayscale
    class RGBToGrayscale(tf.keras.layers.Layer):
        def call(self, inputs):
            # Use TensorFlow's built-in function to convert RGB to grayscale
            return tf.image.rgb_to_grayscale(inputs)

    def __init__(self):
        self.PROJECT_FILES_ROOT = "."
        self.IMG_W = 180
        self.IMG_H = 180
        self.BATCH_SIZE = 16
        self.SEED = 122
        self.VAL_SPLIT = 0.2

        self.NUM_CLASSES = 2


        self.detector = MTCNN()

        self.labels = None
        self.pics_count = None

        self._model = None


        # Tracking the box for smoothness
        self.prev_boxes = {}  # Store previous box positions
        self.smoothing_factor = 0.7  # Adjust this value between 0 and 1 (higher = smoother but more lag)
        self.confidence_threshold = 0.6  # Minimum confidence to change prediction
        self.prev_predictions = {}  # Store previous predictions

    def load_labels(self):
        self.labels = pd.read_csv(f"{self.PROJECT_FILES_ROOT}/dataset/labels.txt", header=None, names=["label", "yaw", "pitch", "roll"], sep=" ")

        self.pics_count = len(self.labels)

        print("Labels loaded. Total images:", self.pics_count)

    def _separate_images_dir(self):
        for i in range(self.pics_count):
            image_filename = self._get_image_filename(i + 1)
            image_path = f"{self.PROJECT_FILES_ROOT}/dataset/files/{image_filename}"
            try:
                if (self.labels["label"][i] == 1):
                    os.renames(image_path, f"{self.PROJECT_FILES_ROOT}/dataset/files/smile/{image_filename}")
                else:
                    os.renames(image_path, f"{self.PROJECT_FILES_ROOT}/dataset/files/nonsmile/{image_filename}")
            except Exception as e:
                print(e)

    def _get_image_filename(index: int, face: int = None) -> str:
        suffix_index = f"{index}"
        if index < 10:
            suffix_index = f"000{index}"
        elif index < 100:
            suffix_index = f"00{index}"
        elif index < 1000:
            suffix_index = f"0{index}"
        
        if face is not None:
            suffix_index = f"{suffix_index}_{face}"

        return f"file{suffix_index}.jpg"

    
    def detect_and_save_faces(self):
        for i in range(self.pics_count):

            image_filename = self._get_image_filename(i + 1)
            try:
                if (self.labels["label"][i] == 1):
                    label = "smile"
                else:
                    label = "nonsmile"

                image_path = f"{self.PROJECT_FILES_ROOT}/dataset/files/{label}/{image_filename}"
                
                image = cv.imread(image_path)
                detected_faces = self.detector.detect_faces(image)


                # Cropping faces and saving them
                for j, result in enumerate(detected_faces):
                    x, y, w, h = result['box']
                    face = image[y:y+h, x:x+w]

                    face_image_filename = self._get_image_filename(i + 1, j + 1)
                    cv.imwrite(f"{self.PROJECT_FILES_ROOT}/dataset/faces/{label}/{face_image_filename}", face)
                
            except Exception as e:
                print(e)


    def _train_val_split(self):
        self._train_ds = tf.keras.utils.image_dataset_from_directory(
            f"{self.PROJECT_FILES_ROOT}/dataset/faces",
            validation_split=self.VAL_SPLIT,
            subset="training",
            seed=self.SEED,
            image_size=(self.IMG_H, self.IMG_W),
            batch_size=self.BATCH_SIZE
        )

        self._val_ds = tf.keras.utils.image_dataset_from_directory(
            f"{self.PROJECT_FILES_ROOT}/dataset/faces",
            validation_split=self.VAL_SPLIT,
            subset="validation",
            seed=self.SEED,
            image_size=(self.IMG_H, self.IMG_W),
            batch_size=self.BATCH_SIZE
        )

        self.class_names = self._train_ds.class_names


    def _pipeline(self, dataset):
        self.AUTOTUNE = tf.data.AUTOTUNE

        return dataset.cache().prefetch(self.AUTOTUNE)
    
    def _get_model(self, use_saved_model=False):
        if self._model is not None:
            return self._model

        if use_saved_model:
            self._model = tf.keras.models.load_model(f"{self.PROJECT_FILES_ROOT}/best_model.keras", safe_mode=False)
        else:

            base_model = tf.keras.models.load_model(f"{self.PROJECT_FILES_ROOT}/cnn_emotion_detection.h5")

            self._model = base_model

        return self._model
    
    def train(self, save_model=False):
        self._train_val_split()

        train_ds_processed = self._pipeline(self._train_ds.shuffle(200))
        val_ds_processed = self._pipeline(self._val_ds)

        model = self._get_model()

        print(model.summary())

        model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
        
        self.epochs=30

        self.history = model.fit(
            train_ds_processed,
            validation_data=val_ds_processed,
            epochs=self.epochs,
            batch_size=self.BATCH_SIZE
        )

        if save_model:
            model.save(f"{self.PROJECT_FILES_ROOT}/saved_models/smile_detect.keras")
            print("Model saved.")

    def draw_history(self):
        if self.history is None:
            print("No history found. Please train the model first.")
            return

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def predict(self, face_img, use_saved_model=False):
        model = self._get_model(use_saved_model=use_saved_model)

        face = cv.resize(face_img, (48, 48))

        img_array = tf.keras.preprocessing.image.img_to_array(face)
        img_array = tf.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        score = tf.nn.sigmoid(predictions)

        score = score.numpy()[0][0]
        label = "smile" if score > 0.5 else "nonsmile"
        confidence = 100 * (1 - score) if label == "nonsmile" else 100 * score
        
        return label, confidence
    
    def predict_from_camera(self, skip=4):
        cap = cv.VideoCapture(0)

        failed_capture_count = 0
        frame_count = 0

        while True:
            frame_count += 1
            ret, frame = cap.read()

            if failed_capture_count > 10:
                print("Too many failed captures. Exiting ...")
                break

            if not ret:
                failed_capture_count += 1
                print("Can't receive frame (stream end?). Exiting ...")
                continue
            
            failed_capture_count = 0

            # Skip every <skip> frames
            if frame_count % skip != 0:
                continue

            detected_faces = self.detector.detect_faces(frame)

            for idx, result in enumerate(detected_faces):
                x, y, w, h = result['box']
                face = frame[y:y+h, x:x+w]

                label, confidence = self.predict(face, use_saved_model=True)
            
                # Smooth the box coordinates
                x, y, w, h = self.smooth_box((x, y, w, h), idx)

                color = (0, 255, 0) if label == "smile" else (0, 0, 255)  # Green for smile, Red for nonsmile

                frame = cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                frame = cv.putText(frame, f"{label} ({confidence:.2f}%)", 
                                (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def predict_from_video(self, video_path, output_path):
        cap = cv.VideoCapture(video_path)

        # Get the original video properties
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS)

        # Define the codec and create a VideoWriter object
        out = cv.VideoWriter(
            output_path, 
            cv.VideoWriter_fourcc(*'mp4v'),  # Codec ('mp4v' is common for .mp4 files)
            fps, 
            (frame_width, frame_height)
        )

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            detected_faces = self.detector.detect_faces(frame)

            for idx, result in enumerate(detected_faces):
                x, y, w, h = result['box']
                face = frame[y:y+h, x:x+w]

                label, confidence = self.predict(face, use_saved_model=True)

                # Smooth the box coordinates
                x, y, w, h = self.smooth_box((x, y, w, h), idx)

                color = (0, 255, 0) if label == "smile" else (0, 0, 255)  # Green for smile, Red for nonsmile

                frame = cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                frame = cv.putText(frame, f"{label} ({confidence:.2f}%)", 
                                (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            out.write(frame)

        # Release resources
        cap.release()
        out.release()
        cv.destroyAllWindows()

    def predict_image(self, image_path):
        img = cv.imread(image_path)

        detected_faces = self.detector.detect_faces(img)

        for result in detected_faces:
            x, y, w, h = result['box']
            face = img[y:y+h, x:x+w]

            label, confidence = self.predict(face, use_saved_model=True)
            
            color = (0, 255, 0) if label == "smile" else (0, 0, 255)  # Green for smile, Red for nonsmile

            frame = cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            frame = cv.putText(frame, f"{label} ({confidence:.2f}%)", 
                            (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv.imshow('frame', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def smooth_box(self, current_box, face_id, reset=False):
        """Smooth the box coordinates using moving average"""
        if reset or face_id not in self.prev_boxes:
            self.prev_boxes[face_id] = current_box
            return current_box
        
        x, y, w, h = current_box
        prev_x, prev_y, prev_w, prev_h = self.prev_boxes[face_id]
        
        # Calculate smoothed coordinates
        smooth_x = int(self.smoothing_factor * prev_x + (1 - self.smoothing_factor) * x)
        smooth_y = int(self.smoothing_factor * prev_y + (1 - self.smoothing_factor) * y)
        smooth_w = int(self.smoothing_factor * prev_w + (1 - self.smoothing_factor) * w)
        smooth_h = int(self.smoothing_factor * prev_h + (1 - self.smoothing_factor) * h)
        
        smoothed_box = (smooth_x, smooth_y, smooth_w, smooth_h)
        self.prev_boxes[face_id] = smoothed_box
        return smoothed_box

@tf.keras.utils.register_keras_serializable(
    package='custom', name=None
)
class RGBToGrayscale(tf.keras.layers.Layer):
        def __init__(self, name="rgb_to_grayscale", **kwargs):
            super(RGBToGrayscale, self).__init__(name=name, **kwargs)

        def call(self, inputs):
            # Use TensorFlow's built-in function to convert RGB to grayscale
            return tf.image.rgb_to_grayscale(inputs)

if __name__ == "__main__":
    fd = FaceDetection()
    # fd.load_labels()
    # fd.detect_and_save_faces()
    # model = fd._get_model(use_saved_model=True)
    # print(model.summary())
    # fd.train(save_model=True)
    # fd.draw_history()
    fd.predict_from_video("./test/VID20250211132839.mp4", "./test/github3.mp4")
    # fd.predict_from_camera()
    # fd.predict_image("./test/test5.jpeg")
    # fd.predict_from_camera()
    # fd.predict_image("./test2.jpeg")
    
