from PIL import Image, ImageDraw
import cv2
import imutils
import time
import numpy as np
import math
import threading
from playsound import playsound
import ringHelper
import serial

class GenDataNotValid(Exception):
    def __init__(self, message="DataNotValid. Must be le 15"):
        self.message = message
        super().__init__(self.message)

    pass


class MatrixDataNotValid(Exception):
    def __init__(self, message="MatrixDataNotValid. Dimension must be eq 3"):
        self.message = message
        super().__init__(self.message)

    pass


def gen(data=0, outfile="out.png", size=100, dim=8):
    canvas = (size * dim, size * dim)
    matrix = encode(data)
    print(matrix)
    img = Image.new("RGB", canvas, (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            if matrix[j][i] == '1':
                color = "black"
            else:
                color = "white"
            draw.rectangle((j * size, i * size, size + j * size, size + i * size), fill=color)
    img.save(outfile)
    #print("Bin: {0}".format(matrix))


def encode(data=0, dim=8):
    if data > 15:
        raise GenDataNotValid
    rep = '11111111'
    rep += '10000001'
    rep += '10111101'
    rep += '10100101'
    rep += '10100101'
    rep += '10011101'
    rep += '10000001'
    rep += '11111111'
    data = ((int('1100', 2) & data) << (dim*3+1)) | ((int('0011', 2) & data ) << (dim*4+3))
    array = (bin(data | int(rep, 2))).replace("0b", "")
    matrix = [array[i*dim:(i+1)*dim] for i in range(dim)]
    return matrix


def decode(matrix, dim=8):
    if len(matrix) != dim or len(matrix[1]) != dim:
        raise MatrixDataNotValid

    for angle in range(0, 6, 1):
        matrix = list(zip(*matrix))[::-1]
        data = ""
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                data += str(matrix[i][j])

        rep = '11111111'
        rep += '10000001'
        rep += '10111101'
        rep += '10100101'
        rep += '10100101'
        rep += '10011101'
        rep += '10000001'
        rep += '11111111'


        print("Data:{0}\tBit42:{1}".format(list(data[i*8:(i+1)*8] for i in range(8)), data[42]))
        if (int(data, 2) & int(rep, 2)) == int(rep, 2) and (int(data[42]) == 0):
            break
        else:
            # print("Rotate matrix")
            pass
        if angle > 3:
            #print("Matrix not valid")
            return -1

    data = (bin(int(data, 2) - int(rep, 2))).replace("0b", "")
    data = (int(data, 2) >> (dim * 3 + 1)) & int('1100', 2) | (int(data, 2) >> (dim * 4 + 3)) & int('0011', 2)
    return data


def to_bw(img,  threshold=200):
    tf = lambda x: 255 if x > threshold else 0
    return img.convert('L').point(tf, mode='1')


def read(filepath, dim=8):
    img = Image.open(r"{0}".format(filepath))
    return img_to_matrix(img, dim)


def img_to_matrix(img, dim=8):
    img = to_bw(img, 200)
    img = img.resize((100 * dim, 100 * dim))
    matrix = [[0]*dim for i in range(dim)]
    for i in range(0, dim):
        for j in range(0, dim):
            if img.getpixel((j*100+50, i*100+50)) == 0:
                matrix[j][i] = 1
            else:
                matrix[j][i] = 0
    return matrix


def test_encoding():
    for i in range(16):
        gen(i, "{0}.png".format(i))
        matrix = read("{0}.png".format(i))
        if decode(matrix) != i:
            print("FAIL! Data:{0}\tMatrix:{1}".format(decode(matrix), matrix))
        else:
            print("All ok. Data:{0}\tMatrix:{1}".format(decode(matrix), matrix))


def find_qr_to_file_my(filepath, outfile="result.png"):
    img = cv2.imread(filepath)
    data = find_qr_stream(img)
    if data:
        matrix = img_to_matrix(data)
        read_data = decode(matrix)
        print("Data:{0}\tMatrix:{1}".format(read_data, matrix))
    data.save(outfile)


def find_qr_stream_old(img,  templatepath='0.png'):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh0, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # template = cv2.imread(templatepath)

    # w, h = template.shape[:-1]
    # mask = cv2.bitwise_not(mask) #inverse mask color
    # img = cv2.bitwise_not(img) #inverse image color
    # img = cv2.bitwise_and(img,  img,  mask=mask) #and with mask
    # img = cv2.bitwise_not(img) #inverse image again

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #set grayscale image
    thresh1, img = cv2.threshold(gray, 150, 255, 0) #treshold again

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("Number of contours detected:", len(contours))

    needcontours = []
    #find all squares:
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if ratio >= 0.9 and ratio <= 1.1:
                needcontours.append(cnt)
    #find all squares with right sizes:
    contours = []
    for cnt in needcontours:
        for cnt2 in needcontours:
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, w1, h1 = cv2.boundingRect(cnt2)
            r1 = w/w1 / 6 * 8
            r2 = h/h1 / 6 * 8
            if r1 >= 0.9 and r1 <= 1.1 and r2 >= 0.9 and r2 <= 1.1:
                dx = abs((x + w) / 2 - (x1 + w1) / 2)
                dy = abs((y + h) / 2 - (y1 + h1) / 2)
                if dx < 50 and dy < 50:
                    contours.append(cnt2)

    if len(contours) != 0:
        print("Number of contours selected:", len(contours))
        x, y, w, h = cv2.boundingRect(contours[0])
        img_pil = Image.fromarray(img)
        img_pil = img_pil.crop((x, y, x + w, y + h))
        cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (x, y), (x + w, y + h), (255, 0, 0), 10)
    else:
        img_pil = None

    return img_pil, img
    #img_pil.save(outfile)


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


def find_qr_stream(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #set grayscale image
    thresh1, img = cv2.threshold(gray, 150, 255, 0) #treshold again

    contours = find_qr_contours(img) #find couturs QR

    if len(contours) != 0:
        #print("Number of contours selected:", len(contours))
        #x, y, w, h = cv2.boundingRect(contours[0])
        pts_src = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)
        #img_pil = Image.fromarray(img)
        #img_pil = img_pil.crop((x, y, x + w, y + h))
        img_qr = crop_np_img(img, pts_src)
        #cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (x, y), (x + w, y + h), (150, 0, 0), 10)
        #cnt = find_qr_contours(np.array(img_pil))
        cnt = find_qr_contours(img_qr)
        if len(cnt) == 0:
            img_pil = None
        else:
            pts_src = cv2.approxPolyDP(contours[0], 0.01 * cv2.arcLength(contours[0], True), True)
            #print(pts_src[0][0])

            pts_src_d = int(math.sqrt((pts_src[0][0][0]-pts_src[1][0][0])**2 + (pts_src[1][0][0]-pts_src[1][0][1])**2))
            #pts_dst = np.array([[0, 0], [0, pts_src_d], [pts_src_d, pts_src_d], [pts_src_d, 0]])
            pts_dst = np.array([[pts_src_d, 0], [pts_src_d, pts_src_d], [0, pts_src_d], [0, 0]])
            h, status = cv2.findHomography(pts_src, pts_dst)
            #im_out = cv2.warpPerspective(np.array(img_pil), h, (pts_src_d, pts_src_d))
            #print(pts_src)
            #print(pts_dst)
            im_out = cv2.warpPerspective(img, h, (pts_src_d, pts_src_d))
            #im_out = cv2.flip(im_out, 1)
            img_pil = Image.fromarray(im_out)
            img_pil = img_pil.crop((0, 0, pts_src_d, pts_src_d))
            #print("x:{0}; y:{1}; w:{2}; h:{3}".format(x, y , w, h))

    else:
        img_pil = None

    return img_pil, img
    #img_pil.save(outfile)


def draw_grid(img):
    w, h = 800, 800
    img = img.resize((w, h))
    draw = ImageDraw.Draw(img)
    for x in range(int(w/16), w, int(w/8)):
        for y in range(int(h/16), h, int(h/8)):
            draw.ellipse((x-5, y-5, x+5, y+5), fill="red")
    return img


def find_qr_contours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    needcontours = []
    #find all squares:
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if ratio >= 0.8 and ratio <= 1.2:
                needcontours.append(cnt)
    #find all squares with right sizes:
    contours = []
    for cnt in needcontours:
        for cnt2 in needcontours:
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, w1, h1 = cv2.boundingRect(cnt2)
            r1 = w/w1 / 6 * 8
            r2 = h/h1 / 6 * 8
            if r1 >= 0.9 and r1 <= 1.1 and r2 >= 0.9 and r2 <= 1.1:
                dx = abs((x + w) / 2 - (x1 + w1) / 2)
                dy = abs((y + h) / 2 - (y1 + h1) / 2)
                if dx < 50 and dy < 50:
                    contours.append(cnt2)
    return contours


def play_sound_qr(sound='Sound.mp3'):
    threading.Thread(target=playsound, args=(sound,), daemon=True).start()


def ring_pass(ser):
    ser.write(bytes("CGCG", 'utf-8'))
    time.sleep(2)
    ser.write(bytes("C0C0", 'utf-8'))


def camera():
    cap = cv2.VideoCapture(0)
    ser = serial.Serial(ringHelper.get_serial(), 115200)
    while True:
        _, img = cap.read()
        # detect and decode
        data, frame = find_qr_stream(img) # check if there is a QRCode in the image
        #time.sleep(.5)
        cv2.imshow("FrameOrig", imutils.resize(frame, width=300))
        if data:
            matrix = img_to_matrix(data)
            read_data = decode(matrix)
            cv2.imshow("Frame", imutils.resize(np.array(draw_grid(data)), width=300))
            if read_data == -1:
                play_sound_qr('wrong.mp3')
                ser.write(bytes("CRCR", 'utf-8'))
                print("Matrix not valid")
            else:
                play_sound_qr('Sound.mp3')
                #ser.write(bytes("CGCG", 'utf-8'))
                print("Data:{0}".format(read_data))
                ring_pass(ser)
            # print("Data:{0}\tMatrix:{1}".format(read_data, matrix))
            time.sleep(.5)
            #data.save("test.png")
            #break
        if cv2.waitKey(1) == ord("q"):
            break
    ser.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    camera()
    #find_qr_to_file_my("in3.png")
    #test_encoding()
