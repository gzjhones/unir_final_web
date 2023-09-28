# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import os
import pickle

from flask import render_template, redirect, request, url_for, jsonify
from flask_dance.contrib.github import github
from flask_login import (
    current_user,
    login_user,
    logout_user
)

from apps import db, login_manager
from apps.authentication import blueprint
from apps.authentication.forms import LoginForm, CreateAccountForm
from apps.authentication.models import Users
from apps.authentication.util import verify_pass

from skimage.morphology import binary_erosion, rectangle, binary_dilation, disk
import cv2 as cv
import numpy as np
from io import BytesIO
import base64

model_path = os.path.join(os.path.dirname(__file__), '../ml-model/model.pkl')
model = pickle.load(open(model_path, 'rb'))

@blueprint.route('/')
def route_default():
    return redirect(url_for('authentication_blueprint.login'))


# Login & Registration

@blueprint.route("/github")
def login_github():
    """ Github login """
    if not github.authorized:
        return redirect(url_for("github.login"))

    res = github.get("/user")
    return redirect(url_for('home_blueprint.index'))


@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)
    if 'login' in request.form:

        # read form data
        username = request.form['username']
        password = request.form['password']

        # Locate user
        user = Users.query.filter_by(username=username).first()

        # Check the password
        if user and verify_pass(password, user.password):
            login_user(user)
            return redirect(url_for('authentication_blueprint.route_default'))

        # Something (user or pass) is not ok
        '''return render_template('accounts/login.html',
                               msg='Wrong user or password',
                               form=login_form)'''

    if not current_user.is_authenticated:
        return render_template('accounts/login.html',
                               form=login_form)
    return redirect(url_for('home_blueprint.index'))


@blueprint.route('/register', methods=['GET', 'POST'])
def register():
    create_account_form = CreateAccountForm(request.form)
    if 'register' in request.form:

        username = request.form['username']
        email = request.form['email']

        # Check usename exists
        user = Users.query.filter_by(username=username).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Username already registered',
                                   success=False,
                                   form=create_account_form)

        # Check email exists
        user = Users.query.filter_by(email=email).first()
        if user:
            return render_template('accounts/register.html',
                                   msg='Email already registered',
                                   success=False,
                                   form=create_account_form)

        # else we can create the user
        user = Users(**request.form)
        db.session.add(user)
        db.session.commit()

        # Delete user from session
        logout_user()

        return render_template('accounts/register.html',
                               msg='Account created successfully.',
                               success=True,
                               form=create_account_form)

    else:
        return render_template('accounts/register.html', form=create_account_form)


# Modelo de predicción
@blueprint.route("/predict", methods=['GET', 'POST'])
def predict():
    rooms = int(request.form['rooms'])
    distance = int(request.form['distance'])
    prediction = model.predict([[rooms, distance]])
    output = round(prediction[0], 2)
    return render_template('home/predict-model.html',
                           prediction_text=f'A house with {rooms} rooms per dwelling and located {distance} km to employment centers has a value of ${output}K')


# Modelo de predicción

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


# Ruta para procesar la imagen y detectar blobs
# @blueprint.route('/procesar_imagen', methods=['POST'])
def procesar_imagen2():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se ha seleccionado ninguna imagen.'})

    imagen = request.files['imagen']

    if imagen.filename == '' or not imagen.filename.endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'El archivo seleccionado no es una imagen válida.'})

    try:
        image_bytes = BytesIO(imagen.read())
        image_np = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        img = cv.imdecode(image_np, cv.IMREAD_COLOR)

        binary_dilation = process_and_detect_blobs(img)

        # Asegúrate de que la imagen sea de 8 bits y en escala de grises
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

        # Devuelve una respuesta JSON con los datos de la imagen resultado
        response_data = {'num_blobs': number_of_blobs, 'image_base64': image_base64}
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'})


@blueprint.route('/procesar_imagen', methods=['POST'])
def procesar_imagen():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se ha seleccionado ninguna imagen.'})

    imagen = request.files['imagen']

    if imagen.filename == '' or not imagen.filename.endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'El archivo seleccionado no es una imagen válida.'})

    try:
        image_bytes = BytesIO(imagen.read())
        image_np = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        img = cv.imdecode(image_np, cv.IMREAD_COLOR)

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, img_threshold = cv.threshold(img_gray, 190, 255, cv.THRESH_BINARY)

        des = cv.bitwise_not(img_threshold)

        contour, hier = cv.findContours(des, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv.drawContours(des, [cnt], 0, 255, -1)

        drawContours = cv.bitwise_not(des)

        image_blur = cv.blur(np.float32(drawContours), (3, 3), 5)

        binary_erosion_s = binary_erosion(image_blur, rectangle(3, 3))

        binary_dilation_s = binary_dilation(binary_erosion_s, disk(1))

        image = np.uint8(255 * binary_dilation_s)

        # image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

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

        keypoints = detector.detect(image)

        blank = np.zeros((1, 1))
        blobs = cv.drawKeypoints(image, keypoints, blank, (255, 0, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        number_of_blobs = len(keypoints)

        retval, buffer = cv.imencode('.png', blobs)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Devuelve una respuesta JSON con los datos de la imagen resultado
        response_data = {'num_blobs': number_of_blobs, 'image_base64': image_base64}
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'})


@blueprint.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('authentication_blueprint.login'))


# Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('home/page-404.html'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('home/page-500.html'), 500
