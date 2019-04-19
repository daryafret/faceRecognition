import os
import random
import cv2
import ctypes as C
import hashlib
import numpy as np
from flask import Flask, render_template, request, send_from_directory, jsonify

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = 'images'
INPUT_NAME = 'file'

face_recognition = C.cdll.LoadLibrary('libface_recognition.so')

def recognizeFaces(image):
    (rows, cols, depth) = (image.shape[0], image.shape[1], image.shape[2])
    detection_results = np.zeros(dtype=np.uint8, shape=(rows, cols, depth))
    recognition_results = np.zeros(dtype=np.uint8, shape=(rows, cols, depth))
    face_recognition.recognizeFaces(image.ctypes.data_as(C.POINTER(C.c_ubyte)), rows, cols,
                                    detection_results.ctypes.data_as(C.POINTER(C.c_ubyte)),
                                    recognition_results.ctypes.data_as(C.POINTER(C.c_ubyte)))
    return detection_results, recognition_results


def process_origin_image(img, filename_n):
    target = os.path.join(APP_ROOT, IMAGE_FOLDER)
    destination = "/".join([target, filename_n])
    img.save(destination)


def process(file_image):
    image_path = IMAGE_FOLDER + file_image.filename
    root_name = hashlib.md5(image_path.encode('utf-8')).hexdigest() + str(random.randint(0, 10000))
    path_root_name = IMAGE_FOLDER + "/" + root_name
    process_origin_image(file_image, root_name + '-0.jpg')
    image = cv2.imread(path_root_name + '-0.jpg', 1)
#    cv2.imshow('Source image', image)
    #detects, recogns = recognizeFaces(image);
    results = []
    results = recognizeFaces(image)
 #   cv2.imshow('Source image', image)
#    cv2.imshow('Detected faces', detects)
 #   cv2.imshow('Recognized faces', recogns)
 #   cv2.waitKey()
    cv2.imwrite(path_root_name + '-1.jpg', results[0])
    cv2.imwrite(path_root_name + '-2.jpg', results[1])
    source_mini_imgs = []
    rot_mini_imgs = []
    print(len(results))
#    for i in range(2):
#      source_img_name = path_root_name + '-3' + str(i) + '.jpg'
#      rotation_img_name = path_root_name + '-3' + str(i) + 'r.jpg'
#      cv2.imwrite(source_img_name, results[0][i])
#      cv2.imwrite(rotation_img_name, results[1][i])
#      source_mini_imgs.append(source_img_name)
#      rot_mini_imgs.append(rotation_img_name)
#    timeMetric = results[3]
    return {'imgs': [path_root_name + '-0.jpg', path_root_name + '-1.jpg', path_root_name + '-2.jpg',
                     path_root_name + '-0.jpg',
                     path_root_name + '-2.jpg'],
            'kl': [path_root_name + '-0.jpg', path_root_name + '-2.jpg'],
            'rot': [path_root_name + '-0.jpg', path_root_name + '-2.jpg'],
            'metrics': {'time': 10, 'test':'test'}}
#            'kl': source_mini_imgs}#,
#            'rot': rot_mini_imgs}#,
#            'metrics': {'time': timeMetric, 'test': 'test'}}



@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if len(request.files.getlist(INPUT_NAME)) != 1:
            return []
        file = request.files[INPUT_NAME]
        result = process(file)
        return jsonify(result)
    return render_template("index.html")


@app.route('/images/<filename>', methods=['GET'])
def send_image(filename):
    return send_from_directory("images", filename)


if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=1234, debug=True) prod
    app.run(port=1234, debug=True)
