from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os
import cv2
import sys

# folder_current = 'D:/Develop/Code_penerapan_AI'
folder_current = 'C:/Users/natha/AndroidStudioProjects/Proyek-Akhir'

sys.path.append(folder_current)
from backend_classification.test_classification import ImageClassifierTester
from backend_classification.train_classification import ImageClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = folder_current + '/server/uploads_images'  # Sesuaikan dengan lokasi penyimpanan Anda
app.config['PREPROCESS_FOLDER'] = folder_current + '/server/preprocess_images'  # Folder untuk hasil pengolahan citra

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    filename = secure_filename(file.filename)
    file_extension = os.path.splitext(filename)[1].lower()
    
    if file_extension != '.jpg' and file_extension != '.jpeg' and file_extension != '.png':
        return jsonify({'error': 'Citra harus dalam format JPG.'})
    else:   
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Proses citra menggunakan OpenCV atau library lain
        # image = cv2.imread(file_path)
        # Tambahkan kode pemrosesan citra di sini

        # processed_image = process_image(image)  # Asumsi process_image adalah fungsi Anda untuk memproses citra
        prediction, features, image = processed_image(file_path)
        # prediction, features, image = train_image(file_path)

        # Simpan citra yang telah diproses
        processed_filename = 'processed_' + filename
        processed_file_path = os.path.join(app.config['PREPROCESS_FOLDER'], processed_filename)
        cv2.imwrite(processed_file_path, image)

        return jsonify(prediction)  # Mengirimkan hasil prediksi ke klien dalam format JSON

def processed_image(file_path):
    try :
        MODEL_DIR = folder_current + '/backend_classification/model'
        FEATURE_DIR =  folder_current + '/backend_classification/fitur'
        FEATURE_TYPE = 'histogram'  # choose from 'histogram', 'glcm', or 'histogram_glcm'
        CLASSIFIER_TYPE = "naive_bayes"  # "mlp", "naive_bayes"
        PCD_TYPE = "filter_gausian"

        TEST_IMAGE_PATH = file_path

        # Create an instance of ImageClassifierTester
        tester = ImageClassifierTester(MODEL_DIR, FEATURE_DIR, FEATURE_TYPE)
        tester.load_data()
        tester.load_classifier(CLASSIFIER_TYPE)

        # Test the classifier on the test image
        prediction, features, image = tester.test_classifier(TEST_IMAGE_PATH)
        print("Prediction:", prediction)
        return prediction, features, image
    except Exception as e:
        print(f"Error in processed_image: {e}")
        raise

if __name__ == '__main__':
    app.run(host='192.168.100.3', port=5000)
