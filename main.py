from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2
import qrcode

# def main():
#     while True:
#         vs = VideoStream(src=0).start()
#         time.sleep(2.0)
#         fps = FPS().start()
#         frame = vs.read()
#         #qframe = imutils.resize(frame, width=400)
#         #(h, w) = frame.shape[:2]
#         #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
#         cv2.imshow("Frame", frame)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
#         fps.update()
#     fps.stop()
#     cv2.destroyAllWindows()
#     vs.stop()
#
# def main2():
#     cap = cv2.VideoCapture(0)
#     detector = cv2.QRCodeDetector()
#     while True:
#         _, img = cap.read()
#         # detect and decode
#         data, bbox, _ = detector.detectAndDecode(img)
#         # check if there is a QRCode in the image
#         if data:
#             a = data
#             break
#         cv2.imshow("QRCODEscanner", img)
#         if cv2.waitKey(1) == ord("q"):
#             break
#         print("data:{0}".format(a))
#     cap.release()
#     cv2.destroyAllWindows()
#
# def Generate_QR(data="drone1", img_name="out.png"):
#     qr = qrcode.QRCode(version=1,
#                        box_size=1,
#                        border=1)
#     qr.add_data(data)
#     qr.make(fit=True)
#     img = qr.make_image(fill_color='black',
#                         back_color='white')
#     img.save(img_name)
#     return img
#
# def Generate_DroneQR(data=0, img_name="out.png"):
#     data


def main():
    while True:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        fps = FPS().start()
        frame = vs.read()
        #qframe = imutils.resize(frame, width=400)
        #(h, w) = frame.shape[:2]
        #blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        fps.update()
    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    #Generate_QR("D", "Denis.png")
    #main2q()
