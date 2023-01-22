import math
import os
import os.path
import time
import threading
from playsound import playsound

import cv2
import numpy as np
from PIL import Image, ImageDraw  # only for creation test images!

import exhelper as ex
import prettyPrint as pp


class QRImage:
    @staticmethod
    def to_bw(img, threshold=150):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # set grayscale image
        thresh1, img = cv2.threshold(gray, threshold, 255, 0)  # threshold it
        return img

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
        needcontours = []
        # find all squares:
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                ratio = float(w) / h
                if 0.8 <= ratio <= 1.2:
                    needcontours.append(cnt)
        # find all squares with right sizes:
        contours = []
        for cnt in needcontours:
            for cnt2 in needcontours:
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
    def find_qr_code(img):
        img = QRImage.to_bw(img, 150)
        contours = QRImage.find_square_contours(img)  # find contours QR

        img_out = None
        if len(contours) != 0:
            # self.log.print("Number of contours selected:", len(contours))
            pts_src = QRImage.get_contour_points(contours[0])
            img_qr = QRImage.crop_np_img(img, pts_src)
            cnt = QRImage.find_square_contours(img_qr)  # this contours should be contains inner contours
            if len(cnt) != 0:
                pts_src = QRImage.get_contour_points(contours[0])
                img_out = QRImage.warp_perspective(pts_src, img)

        return img_out


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

    dim = 8  # matrix dimension

    def __init__(self):
        self.log.print("QRCode object has been created")

    def __del__(self):
        self.log.print("QRCode object has been deleted")

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

            # self.log.print("Data:{0}\tBit42:{1}".format(list(data[i * 8:(i + 1) * 8] for i in range(8)), data[42]))
            if (int(data, 2) & int(self.rep, 2)) == int(self.rep, 2) and (int(data[42]) == 0):
                self.log.print("Data:{0}\tBit42:{1}".format(list(data[i * 8:(i + 1) * 8] for i in range(8)), data[42]))
                break
            else:
                # print("Rotate matrix")
                pass
            if angle > 3:
                # print("Matrix not valid")
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
        img = Image.open(r"{0}".format(filepath))
        img = QRImage.to_bw(np.array(img), 150)
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
            self.save_image(data=i, outfile="{0}/{1}.png".format(folder, i))
            matrix = self.read_image("{0}/{1}.png".format(folder, i))
            if self.decode(matrix) != i:
                print("FAIL! Data:{0}\tMatrix:{1}".format(self.decode(matrix), matrix))
            else:
                print("All ok. Data:{0}\tMatrix:{1}".format(self.decode(matrix), matrix))


# class QRStreamReader:
#     log = pp.Log("QRStreamReader")
#
#     def __init__(self, camera_id=0):
#         self.CameraID = camera_id
#         self.QRCode = QRCode()
#         self.log.print("QRStreamReader with camera id {0} has been created".format(self.CameraID))
#
#     # def find_in_file(self, filepath, outfile="result.png"):
#     #     img = cv2.imread(filepath)
#     #     data = QRImage.find_qr_code(img)
#     #     if data:
#     #         matrix = QRImage.find_qr_code(data)
#     #         read_data = QRCode().decode(matrix)
#     #         print("Data:{0}\tMatrix:{1}".format(read_data, matrix))
#     #     data.save(outfile)


class QREvents:
    @staticmethod
    def play_sound_qr(sound='Sound.mp3'):
        threading.Thread(target=playsound, args=(sound,), daemon=True).start()

    @staticmethod
    def ring_pass(ser):
        ser.write(bytes("CGCG", 'utf-8'))
        time.sleep(2)
        ser.write(bytes("C0C0", 'utf-8'))

    @staticmethod
    def ring_fail(ser):
        ser.write(bytes("CRCR", 'utf-8'))

    @staticmethod
    def ring_pass_async(ser):
        threading.Thread(target=QREvents.ring_pass, args=(ser,), daemon=True).start()


if __name__ == '__main__':
    QRCode = QRCode()
    QRCode.test_encoding('test')
    # main()
