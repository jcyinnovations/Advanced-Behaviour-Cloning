import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

#########################################################
# ImageProcessing library for preprocessing data
#########################################################
from ImageProcessing import get_viewport, image_gradient, data_normalization

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

options = np.array([x for x in range(51)])
prev_angle=0
#########################################################
# Find the steering angle based on the model predictions
#########################################################
def get_steering_angle(prediction):
    global prev_angle
    angle = prev_angle
    probability = np.max(prediction)
    #choice = np.argmax(prediction)
    choice = np.sum(options * prediction[0]) #Expectation not THE prediction
    angle = (choice-25)/25
    prev_angle = angle
    return angle


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        
        #################################
        # Image preprocessing 
        #################################
        image_array = get_viewport(image_array, size=(40,40))
        image_array = image_array/255 - 0.5 #preprocess_image(image)   
        
        prediction = model.predict(image_array[None, :, :, :], batch_size=1)
        
        ##########################################
        # Steering angles converted to 51 classes 
        ##########################################
        #steering_angle = get_steering_angle(prediction)
        steering_angle = prediction[0][0]
		
        min_speed = 15
        max_speed = 20
        if float(speed) < min_speed:
            throttle = 0.5
        elif float(speed) > max_speed:
            throttle = -1.0
        else:
            throttle = 0.1
        
        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
            print("{0}.jpg,{1},{2}".format(image_filename, steering_angle, throttle), file=logfile)
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    logfile = None
    
    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
            ', but the model was built using ', model_version)
        
    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
		
		#Open the steering logfile
        logfile = open('steering.csv', 'w')
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    #Close the logfile
    if logfile is not None:
        logfile.close()
