import math
import os
import os.path
import time
import threading
# import queue
from playsound import playsound

import cv2
import numpy as np
from PIL import Image, ImageDraw  # only for creation test images!

import exhelper as ex
import prettyPrint as pp


class ImgHelper:
    @staticmethod
    def square(size=300):
        image = np.zeros((size, size, 3), np.uint8)
        return image

    @staticmethod
    def to_bw(img, threshold=150):
        # img = ImgHelper.increase_contrast(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # set grayscale image
        thresh1, img = cv2.threshold(gray, threshold, 255, 0)  # threshold it
        return img

    @staticmethod
    def increase_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8)):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)  # split to channels
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)  # creating CLAHE
        cl = clahe.apply(l_channel)  # Applying CLAHE to L-channel
        l_img = cv2.merge((cl, a, b))  # merge the CLAHE enhanced L-channel with the a and b channel
        enhanced_img = cv2.cvtColor(l_img, cv2.COLOR_LAB2BGR)  # convert back to BGR

        return enhanced_img

    @staticmethod
    def crop_np_img(img, nparray):
        rect = cv2.boundingRect(nparray)
        x, y, w, h = rect
        cropped = img[y:y + h, x:x + w].copy()
        nparray = nparray - nparray.min(axis=0)

        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [nparray], -1, (255, 255, 255), -1, cv2.LINE_AA)

        dst = cv2.bitwise_and(cropped, cropped, mask=mask)

        bg = np.ones_like(cropped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)
        dst2 = bg + dst

        return dst2

    @staticmethod
    def find_square_contours(img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        need_contours = []
        # find all squares:
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                ratio = float(w) / h
                if 0.8 <= ratio <= 1.2:
                    need_contours.append(cnt)
        # find all squares with right sizes:
        contours = []
        for cnt in need_contours:
            for cnt2 in need_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x1, y1, w1, h1 = cv2.boundingRect(cnt2)
                r1 = w / w1 / 6 * 8
                r2 = h / h1 / 6 * 8
                if 0.9 <= r1 <= 1.1 and 0.9 <= r2 <= 1.1:
                    dx = abs((x + w) / 2 - (x1 + w1) / 2)
                    dy = abs((y + h) / 2 - (y1 + h1) / 2)
                    if dx < 50 and dy < 50:  # contours should be axial
                        contours.append(cnt2)
        return contours

    @staticmethod
    def get_contour_points(contour):
        return cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    @staticmethod
    def warp_perspective(pts_src, img):
        pts_src_d = int(
            math.sqrt((pts_src[0][0][0] - pts_src[1][0][0]) ** 2 + (pts_src[1][0][0] - pts_src[1][0][1]) ** 2))
        pts_dst = np.array([[pts_src_d, 0], [pts_src_d, pts_src_d], [0, pts_src_d], [0, 0]])
        h, status = cv2.findHomography(pts_src, pts_dst)
        return cv2.warpPerspective(img, h, (pts_src_d, pts_src_d))

    @staticmethod
    def find_qr_code(img, threshold=150):
        img = ImgHelper.to_bw(img, threshold)
        contours = ImgHelper.find_square_contours(img)  # find contours QR

        img_out = None
        if len(contours) != 0:
            # self.log.print(f"Number of contours selected:{len(contours)}")
            pts_src = ImgHelper.get_contour_points(contours[0])
            img_qr = ImgHelper.crop_np_img(img, pts_src)
            cnt = ImgHelper.find_square_contours(img_qr)  # this contours should be contains inner contours
            if len(cnt) != 0:
                pts_src = ImgHelper.get_contour_points(contours[0])
                img_out = ImgHelper.warp_perspective(pts_src, img)

        return img_out, img


class QRCode:
    log = pp.Log("QRCode")

    # base matrix for QR:
    rep = '11111111'
    rep += '10000001'
    rep += '10111101'
    rep += '10100101'
    rep += '10100101'
    rep += '10011101'
    rep += '10000001'
    rep += '11111111'

    def __init__(self, dim=8, bits=4):
        self.log.print("QRCode object has been created")
        self.dim = dim  # matrix dimension
        self.bits = bits  # number of info bits
        self.validate_dim()

    def validate_dim(self):
        if self.dim != 8 or self.bits != 4:
            raise ex.MatrixDimensionNotValid

    def __del__(self):
        self.log.print("QRCode object has been deleted")

    def set_base_matrix(self, base_matrix):
        self.log.print("Base matrix has been set")
        self.rep = base_matrix

    def encode(self, data):
        if data > 15:
            raise ex.GenDataNotValid

        data = ((int('1100', 2) & data) << (self.dim * 3 + 1)) | \
               ((int('0011', 2) & data) << (self.dim * 4 + 3))
        array = (bin(data | int(self.rep, 2))).replace("0b", "")
        matrix = [array[i * self.dim:(i + 1) * self.dim] for i in range(self.dim)]
        return matrix

    def decode(self, matrix):
        if len(matrix) != self.dim or len(matrix[1]) != self.dim:
            raise ex.MatrixDataNotValid

        for angle in range(0, 6, 1):
            matrix = list(zip(*matrix))[::-1]
            data = ""
            for i in range(0, len(matrix)):
                for j in range(0, len(matrix[i])):
                    data += str(matrix[i][j])

            if (int(data, 2) & int(self.rep, 2)) == int(self.rep, 2) and (int(data[42]) == 0):
                matrix_bin = list(data[i * 8:(i + 1) * 8] for i in range(8))
                # self.log.print(f"Matrix:{matrix_bin}\tBit42:{data[42]}")
                break
            else:
                # self.log.print("Rotate matrix")
                pass
            if angle > 3:
                # self.log.print("Matrix not valid")
                return -1

        data = (bin(int(data, 2) - int(self.rep, 2))).replace("0b", "")
        data = (int(data, 2) >> (self.dim * 3 + 1)) & int('1100', 2) | \
               (int(data, 2) >> (self.dim * 4 + 3)) & int('0011', 2)
        return data

    def save_image(self, data=0, outfile="out.png", size=100):
        canvas = (size * self.dim, size * self.dim)
        matrix = self.encode(data)
        self.log.print(matrix)
        img = Image.new("RGB", canvas, (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                if matrix[j][i] == '1':
                    color = "black"
                else:
                    color = "white"
                draw.rectangle((j * size, i * size, size + j * size, size + i * size), fill=color)
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        img.save(outfile)

    def read_image(self, filepath):
        img = Image.open(rf"{filepath}")
        img = ImgHelper.to_bw(np.array(img), 150)
        return self.np_to_matrix(img)

    def np_to_matrix(self, img):  # this method works with NP array
        img = cv2.resize(img, dsize=(100 * self.dim, 100 * self.dim), interpolation=cv2.INTER_CUBIC)
        matrix = [[0] * self.dim for i in range(self.dim)]
        for i in range(0, self.dim):
            for j in range(0, self.dim):
                if img[j * 100 + 50, i * 100 + 50] == 0:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = 0
        return matrix

    def test_encoding(self, folder):
        for i in range(16):
            test_path = f"{folder}/{i}.png"
            self.save_image(data=i, outfile=test_path)
            matrix = self.read_image(test_path)
            if self.decode(matrix) != i:
                test_result = "Failed!"
            else:
                test_result = "Passed!"
            self.log.print(f"{test_result}\tData:{self.decode(matrix)}\tMatrix:{matrix}")


class QRStreamReader(threading.Thread):
    def __init__(self, camera_id=0, width=300, queue=None, dead_time=0.5):
        self.queue = queue
        self.CameraID = camera_id
        self.log = pp.Log(f"QRStreamReader{self.CameraID}")
        self.QRAnalyzer = QRCode()
        self.cap = cv2.VideoCapture(self.CameraID)
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # set autoexposure
        self.LastQRCode = ImgHelper.square(300)
        self.LastQRData = -1
        self.QRCode = None
        self.QRData = -1
        self.RawImage = None
        self.BWImage = None
        self.img_width = width
        self._should_stop = False
        # self._lock = threading.RLock()
        threading.Thread.__init__(self)
        self.setDaemon(True)  # for right shutting down app
        self.log.print(f"QRStreamReader with camera id {self.CameraID} has been created")
        self.is_locked = False
        self.dead_time = dead_time
        self.threshold = 100

    def __del__(self):
        self.cap.release()
        self.log.print(f"QRStreamReader with camera id {self.CameraID} has been deleted")

    def unlock_analyze(self):
        # self.unlock_timer.cancel()
        self.is_locked = False
        self.log.print("Stream analyzer has been unlocked")

    def lock_analyze(self):
        unlock_timer = threading.Timer(interval=self.dead_time, function=lambda: self.unlock_analyze())
        self.is_locked = True
        self.log.print("Stream analyzer has been locked")
        unlock_timer.start()

    def read_image(self):
        _, self.RawImage = self.cap.read()
        # self.RawImage = ImgHelper.increase_contrast(self.RawImage)
        self.QRCode, self.BWImage = ImgHelper.find_qr_code(self.RawImage,
                                                           self.threshold)  # check if there is a QRCode in the image
        if self.QRCode is not None and not self.is_locked:
            matrix = self.QRAnalyzer.np_to_matrix(self.QRCode)
            self.QRData = self.QRAnalyzer.decode(matrix)
            if self.QRData == -1:
                self.log.print("Matrix not valid")
            else:
                self.log.print(f"Data:{self.QRData}")
                self.lock_analyze()
            if self.queue is not None:
                self.queue.put({
                    'Data': self.QRData,
                    'QRCode': self.QRCode,
                    'CameraID': self.CameraID
                }, block=False)
                self.LastQRCode = self.QRCode
                self.LastQRData = self.QRData
            # time.sleep(self.dead_time)

        pass

    def run(self):
        self.log.print("Stream analyzer has been run")
        try:
            while not self._should_stop:
                time.sleep(0)  # for control from other threads
                self.read_image()
        except (StopIteration, EOFError):
            pass

    def stop(self):
        self._should_stop = True
        while self.is_alive():
            time.sleep(0.001)


class QREvents:
    @staticmethod
    def play_sound_qr(sound='Sound.mp3'):
        threading.Thread(target=playsound, args=(sound,), daemon=True).start()

    @staticmethod
    def ring_pass(ser):
        if ser is not None:
            ser.write(bytes("CGCG", 'utf-8'))
            time.sleep(2)
            ser.write(bytes("C0C0", 'utf-8'))

    @staticmethod
    def ring_fail(ser):
        if ser is not None:
            ser.write(bytes("CRCR", 'utf-8'))

    @staticmethod
    def ring_pass_async(ser):
        if ser is not None:
            threading.Thread(target=QREvents.ring_pass, args=(ser,), daemon=True).start()


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@singleton
class QRSettings:
    log = pp.Log("QRSettings")

    def __init__(self):
        pass

    def config(self,
               sound_ring_passed: str,
               sound_ring_failed: str,
               serial_speed: int,
               camera_id: [int]
               ):
        if not os.path.exists(sound_ring_passed):
            raise FileNotFoundError
        self._sound_ring_passed = sound_ring_passed
        if not os.path.exists(sound_ring_failed):
            raise FileNotFoundError
        self._sound_ring_failed = sound_ring_failed
        self._serial_speed = serial_speed
        self._camera_id = camera_id
        self.log.print(f"Settings: {self.__dict__}")

    @property
    def camera_id(self):
        return self._camera_id

    @property
    def serial_speed(self):
        return self._serial_speed

    @property
    def sound_ring_failed(self):
        return self._sound_ring_failed

    @property
    def sound_ring_passed(self):
        return self._sound_ring_passed


if __name__ == '__main__':
    QRCode = QRCode()
    QRCode.test_encoding('test')
    # main()
