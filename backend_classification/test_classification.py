import os
import cv2
import numpy as np
import pickle
import time

import sys
# folder_current = 'D:/Develop/Code_penerapan_AI'
folder_current = 'C:/Users/natha/AndroidStudioProjects/Proyek-Akhir'
sys.path.append(folder_current + '/backend_classification')
from FeatureExtractor_GLCM import GLCMFeatureExtractor

class ImageClassifierTester:
    def __init__(self, model_dir, feature_dir, feature_type):
        self.model_dir = model_dir
        self.feature_dir = feature_dir
        self.feature_type = feature_type
        self.data = None
        self.labels = None
        self.classifier = None
        self.feature_extractors = {
            "histogram": self.extract_histogram,
            "glcm": self.extract_glcm
        }

    def extract_histogram(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
         # Take a subset of features if you want a fixed number (e.g., first 60)
        hist = hist[:60]
        # Flatten and reshape histogram to 1-dimensional array
        hist = hist.reshape(1, -1)
        return hist

    def extract_glcm(self, image):
        feature_extractor = GLCMFeatureExtractor()
        glcm_features = feature_extractor.compute_glcm_features(image)
        return glcm_features

    def load_data(self):
        self.data = np.load(os.path.join(self.feature_dir, 'data.npy'))
        self.labels = np.load(os.path.join(self.feature_dir, 'labels.npy'))

    def load_classifier(self, classifier_type):
        model_file = os.path.join(self.model_dir, f'{classifier_type}_model.pkl')
        with open(model_file, 'rb') as f:
            self.classifier = pickle.load(f)

    def read_image(self, test_image_path):
        image = cv2.imread(test_image_path)
        return image
    
    def equalize_color_image(self, image):
        if image is None:
            print("Error: Cannot read image.")
            return
        # Equalize histogram pada channel Red (0), Green (1), dan Blue (2)
        image_equalized = image.copy()
        for i in range(3):
            image_equalized[:,:,i] = cv2.equalizeHist(image[:,:,i])
        return image_equalized


    def process_image(self, image, namafile):
        image = self.equalize_color_image(image)
        
        # Simpan citra yang telah diproses
        processed_filename = 'processed_' + namafile
        # alamat = 'D:/Develop/Code_penerapan_AI/backend_classification/preprocessing_test'
        alamat = 'C:/Users/natha/AndroidStudioProjects/Proyek-Akhir/backend_classification/preprocessing_test'
        processed_file_path = os.path.join(alamat, processed_filename)
        cv2.imwrite(processed_file_path, image)
        return image

    def test_classifier(self, test_image_path):
        image = self.read_image(test_image_path)

        namafile = f"processed_{int(time.time())}.jpg" # penambahan

        image = self.process_image(image, namafile) # kurang 1 parameter yaitu namafile
        features = self.feature_extractors[self.feature_type](image)
        features = features.reshape(1, -1)

        print("Number of features:", len(features[0]))

        prediction = self.classifier.predict(features)
        return prediction[0], features, image


if __name__ == "__main__":
    folder_current = 'C:/Users/natha/AndroidStudioProjects/Proyek-Akhir'
    MODEL_DIR = folder_current + '/backend_classification/model'
    FEATURE_DIR = folder_current + '/backend_classification/fitur'
    FEATURE_TYPE = 'histogram'  # choose from 'histogram', 'glcm', or 'histogram_glcm'
    CLASSIFIER_TYPE = "naive_bayes"  # "mlp", "naive_bayes"

    # TEST_IMAGE_PATH = folder_current + '/backend_classification/dataset/Car_lite/Convertible/convertible_0_09062020_145449.jpg'
    TEST_IMAGE_PATH = folder_current + '/backend_classification/dataset/Car_lite/Pickup/Pickup_4_09062020_215826.jpg'

    # Create an instance of ImageClassifierTester
    tester = ImageClassifierTester(MODEL_DIR, FEATURE_DIR, FEATURE_TYPE)
    tester.load_data()
    tester.load_classifier(CLASSIFIER_TYPE)

    # Test the classifier on the test image
    prediction = tester.test_classifier(TEST_IMAGE_PATH)
    print("Prediction:", prediction)
