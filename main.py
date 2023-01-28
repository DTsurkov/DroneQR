import time

import cv2
import imutils

import serial

import ringHelper
import qrutils
import prettyPrint as pp


def main():
    log = pp.Log("Main")
    cap = cv2.VideoCapture(0)
    # ser = serial.Serial(ringHelper.get_serial(), 115200)
    qr = qrutils.QRCode()
    while True:
        _, img = cap.read()
        data = qrutils.ImgHelper.find_qr_code(img)  # check if there is a QRCode in the image
        # time.sleep(.5)
        cv2.imshow("FrameOrig", imutils.resize(img, width=300))
        if data is not None:
            matrix = qr.np_to_matrix(data)
            read_data = qr.decode(matrix)
            cv2.imshow("Frame", imutils.resize(data, width=300))
            if read_data == -1:
                qrutils.QREvents.play_sound_qr('wrong.mp3')
                # qrutils.QREvents.ring_fail(ser)
                log.print("Matrix not valid")
            else:
                qrutils.QREvents.play_sound_qr('Sound.mp3')
                log.print(f"Data:{read_data}")
                # qrutils.QREvents.ring_pass_async(ser)
                time.sleep(.5)
        if cv2.waitKey(1) == ord("q"):
            break
    # ser.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
