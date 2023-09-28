# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import Flask, render_template, request, jsonify, send_file, make_response
from flask_cors import CORS, cross_origin
from flask_login import login_required
from jinja2 import TemplateNotFound
from PIL import Image
import io

from skimage.morphology import binary_dilation, binary_erosion, erosion, rectangle, disk
from skimage.color import rgb2gray
from skimage.io import imread
import cv2 as cv
import numpy as np
import requests
from io import BytesIO
import base64
import pybase64


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


CORS(blueprint, resources={r"/api/api_process_image": {"origins": "*"}})


@blueprint.route('/api/api_process_image', methods=['POST'])
def api_process_image():
    json_request = request.get_json()
    if 'image' not in json_request:
        response_data = {'error': 'Wrong request data'}
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response

    try:
        image_request = Image.open(io.BytesIO(base64.decodebytes(bytes(json_request['image'], "utf-8"))))
        image_request.save('./image_request.jpeg')
        img_gray = cv.cvtColor(cv.imread('./image_request.jpeg'), cv.COLOR_BGR2GRAY)

        blobs, white, red = define_results(filter_image(img_gray, False), filter_image(img_gray, True))

        image_response = pybase64.b64encode((open("response_globes.jpeg", "rb")).read())

        response = jsonify({'all': blobs, 'white': white, 'red': red, 'image': image_response.decode('utf-8')})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    except Exception as e:
        response = jsonify({'error': repr(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response


def filter_image(img_gray, is_white_globes):
    ret, img_threshold = cv.threshold(img_gray, 100 if is_white_globes else 190, 255, cv.THRESH_BINARY)
    des = cv.bitwise_not(img_threshold)
    contour, hier = cv.findContours(des, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv.drawContours(des, [cnt], 0, 255, -1)
    draw_contours = cv.bitwise_not(des)

    if is_white_globes:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        res = cv.morphologyEx(draw_contours, cv.MORPH_OPEN, kernel)
        img_median_blur = cv.medianBlur(draw_contours, 1)
        img_binary_dilation = binary_dilation(img_median_blur, disk(2))
        img_erosioned = erosion(img_binary_dilation, disk(5))
        cv.imwrite('./white_globes.jpeg', 255 * img_erosioned)
        return cv.imread('./white_globes.jpeg', 0)
    else:
        image_blur = cv.blur(np.float32(draw_contours), (3, 3), 5)
        img_binary_erosion = binary_erosion(image_blur, rectangle(3, 3))
        img_binary_dilation = binary_dilation(img_binary_erosion, disk(1))
        cv.imwrite('./all_globes.jpeg', 255 * img_binary_dilation)
        return cv.imread('./all_globes.jpeg', 0)


def define_results(img_all_globes, img_white_globes):
    params = cv.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.filterByCircularity = True
    params.minCircularity = 0.01
    params.filterByConvexity = True
    params.minConvexity = 0.1
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    blank = np.zeros((1, 1))
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_white_globes)
    white_blobs_count = len(keypoints)
    keypoints = detector.detect(img_all_globes)
    all_blobs_count = len(keypoints)
    all_blobs = cv.drawKeypoints(img_all_globes, keypoints, blank, (255, 0, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('./response_globes.jpeg', 255 * all_blobs)

    return all_blobs_count, white_blobs_count, all_blobs_count-white_blobs_count
