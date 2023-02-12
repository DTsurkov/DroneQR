import time

import cv2
import imutils
import sys
import serial

import ringHelper
import qrutils
import queue
import prettyPrint as pp


def main():  # it is a test main function. Just for QRcode recognition testing
    # log = pp.Log("Main")
    QSSettings = qrutils.QRSettings(
        sound_ring_passed="Sound.mp3",
        sound_ring_failed="wrong.mp3",
        serial_speed=115200,
        camera_id=[0]
    )
    try:
        ser = serial.Serial(ringHelper.get_serial(), QSSettings.serial_speed)
    except IndexError:
        ser = None

    qr_streams = []
    for camera_id in QSSettings.camera_id:
        qr_streams.append(qrutils.QRStreamReader(camera_id=camera_id, queue=queue.Queue()))

    for qr_stream in qr_streams:
        qr_stream.start()

    while True:
        for qr_stream in qr_streams:
            if qr_stream.RawImage is not None:
                cv2.imshow(f"Stream_{qr_stream.CameraID}", imutils.resize(qr_stream.RawImage, width=300))
            if qr_stream.BWImage is not None:
                cv2.imshow(f"BW_{qr_stream.CameraID}", imutils.resize(qr_stream.BWImage, width=300))
            if not qr_stream.queue.empty():
                data = qr_stream.queue.get()
                cv2.imshow(f"Code_{qr_stream.CameraID}", imutils.resize(data['QRCode'], width=300))
                if data['Data'] == -1:
                    qrutils.QREvents.play_sound_qr(QSSettings.sound_ring_failed)
                    qrutils.QREvents.ring_fail(ser)
                else:
                    qrutils.QREvents.play_sound_qr(QSSettings.sound_ring_passed)
                    qrutils.QREvents.ring_pass_async(ser)
                    # time.sleep(.5)
        if cv2.waitKey(1) == ord("q"):
            for qr_stream in qr_streams:
                qr_stream.stop()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
