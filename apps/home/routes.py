# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import Flask, render_template, request, jsonify, send_file, make_response
from flask_cors import CORS
from flask_login import login_required
from jinja2 import TemplateNotFound
from PIL import Image
import io

from skimage.morphology import binary_erosion, rectangle, binary_dilation, disk
from skimage.color import rgb2gray
from skimage.io import imread
import cv2 as cv
import numpy as np
import requests
from io import BytesIO
import base64

@blueprint.route('/index')
@login_required
def index():

    return render_template('home/index.html', segment='index')


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
    
def process_and_detect_blobs(image):
    # Convierte la imagen a escala de grises si no lo está
    if image.shape[2] == 3:
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        img_gray = image

    ret, img_threshold = cv.threshold(img_gray, 190, 255, cv.THRESH_BINARY)
    des = cv.bitwise_not(img_threshold)
    contour, hier = cv.findContours(des, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv.drawContours(des, [cnt], 0, 255, -1)

    drawContours = cv.bitwise_not(des)
    image_blur = cv.blur(np.float32(drawContours), (3, 3), 5)

    binary_erosion = cv.morphologyEx(image_blur, cv.MORPH_ERODE, np.ones((3, 3), np.uint8))
    binary_dilation = cv.morphologyEx(binary_erosion, cv.MORPH_DILATE, np.ones((3, 3), np.uint8))

    return binary_dilation

CORS(blueprint, resources={r"/api/api_process_image": {"origins": "http://127.0.0.1:5000"}})
@blueprint.route('/api/api_process_image', methods=['POST'])
def api_process_image():
    if 'image' not in request.files:
        response_data = {'error': 'No es una imagen'}
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response
    
    imagen = request.files['image']
    if imagen.filename == '' or not imagen.filename.endswith(('.jpg', '.jpeg', '.png')):
        response_data = {'error': 'El archivo seleccionado no es una imagen válida.'}
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
    
    try:
        image_bytes = BytesIO(imagen.read())
        image_np = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        img = cv.imdecode(image_np, cv.IMREAD_COLOR)

        binary_dilation = process_and_detect_blobs(img)

        if binary_dilation.dtype != np.uint8:
            binary_dilation = np.uint8(binary_dilation)

        image = np.uint8(255 * binary_dilation)

        params = cv.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 20
        # params.maxArea = 2000

        params.filterByCircularity = True
        params.minCircularity = 0.01
        # params.maxCircularity = 1

        params.filterByConvexity = True
        params.minConvexity = 0.1

        params.filterByInertia = True
        params.minInertiaRatio = 0.01

        detector = cv.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_dilation)

        blank = np.zeros((1, 1))
        blobs = cv.drawKeypoints(image, keypoints, blank, (255, 0, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        number_of_blobs = len(keypoints)

        retval, buffer = cv.imencode('.png', blobs)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        response = make_response(send_file(io.BytesIO(buffer), mimetype='image/jpeg'))
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Expires'] = number_of_blobs
        return response
    
    except Exception as e:
        response = jsonify({'error': 'Excepción'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
   
