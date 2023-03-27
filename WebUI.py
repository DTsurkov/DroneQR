import time

import cv2
import imutils
import sys
import serial

import ringHelper
import qrutils
import queue
import prettyPrint as pp

import flask
from datetime import datetime

WebUI = flask.Flask(__name__)


def gen_frames(stream_id, img_type):
    global qr_streams
    qr = qr_streams[stream_id]
    while True:
        frame = getattr(qr, img_type)

        if frame is not None:
            success, image = cv2.imencode('.jpg', frame)

            if success:
                data = image.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')


@WebUI.route('/video_feed/<int:stream_id>_<string:img_type>')
def video_feed(stream_id, img_type):
    return flask.Response(gen_frames(stream_id, img_type),
                          mimetype='multipart/x-mixed-replace; boundary=frame',
                          headers={'Cache-Control': 'no-cache',
                                   'Expires': datetime.utcnow()})


@WebUI.route('/')
def index():
    return flask.render_template('index.html', len=len(qr_streams))


QSSettings = qrutils.QRSettings()

QSSettings.config(
    sound_ring_passed="Sound.mp3",
    sound_ring_failed="wrong.mp3",
    serial_speed=115200,
    camera_id=[0]
    # camera_id=[camera[0] for camera in camHelper.get_cameras("VendorID_6380")]
)

qr_streams = []


def config():
    try:
        ser = serial.Serial(ringHelper.get_serial(), QSSettings.serial_speed)
    except IndexError:
        ser = None

    global qr_streams
    for camera_id in QSSettings.camera_id:
        qr_streams.append(qrutils.QRStreamReader(camera_id=camera_id, queue=queue.Queue()))

    for qr_stream in qr_streams:
        qr_stream.start()


if __name__ == '__main__':
    config()
    WebUI.debug = True
    WebUI.run()  # .run(host='0.0.0.0')
