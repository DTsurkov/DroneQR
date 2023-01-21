from PIL import Image, ImageDraw
import cv2
import numpy as np

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


def gen(data=0, outfile="out.png", size=100):
    canvas = (size * 3, size * 3)
    matrix = encode(data)
    #print(matrix)
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


def encode(data=0, dim = 3):
    if data > 15:
        raise GenDataNotValid
    data = ((int('1100', 2) & data) << 1) | (int('0011', 2) & data)
    array = (bin(data | int('111100100', 2))).replace("0b", "")
    matrix = [array[i * dim:(i + 1) * dim] for i in range(dim)]
    return matrix


def decode(matrix):
    if len(matrix) != 3 or len(matrix[1]) != 3:
        raise MatrixDataNotValid

    data = ""
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            data += str(matrix[i][j])

    data = (bin(int(data,2) - int('111100100', 2))).replace("0b", "")
    data = ((int('11000', 2) & int(data,2)) >> 1) | (int('00011', 2) & int(data,2))
    return data


def to_bw(img,  threshold=200):
    tf = lambda x: 255 if x > threshold else 0
    return img.convert('L').point(tf, mode='1')


def read(filepath):
    img = Image.open(r"{0}".format(filepath))
    bw = to_bw(img, 200)
    bw = bw.resize((300, 300))
    matrix = [ [0]*3 for i in range(3)]
    for i in range(0, 3):
        for j in range(0, 3):
            if bw.getpixel((j*100+50, i*100+50)) == 0:
                matrix[j][i] = 1
            else:
                matrix[j][i] = 0
    return matrix
    #bw.save("C_{0}".format(filepath))


def test_encoding():
    for i in range(16):
        gen(i, "{0}.png".format(i))
        matrix = read("{0}.png".format(i))
        if decode(matrix) != i:
            print("FAIL!")
        else:
            print("All ok. Data:{0}\tMatrix:{1}".format(decode(matrix), matrix))


def find_qr_to_file_old(filepath,  templatepath='0.png',  outfile="result.png",  threshold=.8):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh0, img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    template = cv2.imread(templatepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh1, template = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    w, h = template.shape[:-1]

    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)#, mask=mask)
    #res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    img_pil = Image.fromarray(img)
    #draw = ImageDraw.Draw(img_pil)

    for pt in zip(*loc[::-1]):  # Switch collumns and rows
        #draw.rectangle((pt[0], pt[1], pt[0] + w, pt[1] + h), outline ="red")
        qrcode = img_pil.crop((pt[0], pt[1], pt[0] + w, pt[1] + h))
        #cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), thickness=1)

    qrcode.save(outfile)
    #cv2.imwrite(outfile, img)


def find_qr_to_file_mask(filepath,  templatepath='11.png',  outfile="result.png",  threshold=0.6):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh0, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    template = cv2.imread(templatepath)

    w, h = template.shape[:-1]
    mask = cv2.bitwise_not(mask)
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(img,  img,  mask=mask)
    img = cv2.bitwise_not(img)

    # img_pil = Image.fromarray(img)
    # img_pil.save(outfile)
    #mask = np.zeros(template.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (100, 100), (400, 400), 255, -1)

    #mask = cv2.bitwise_not(mask)
    img_pil = Image.fromarray(img)
    img_pil.save(outfile)


    # res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)#, mask = mask)
    #
    # #res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    # loc = np.where(res >= threshold)
    #
    # img_pil = Image.fromarray(img)
    # #draw = ImageDraw.Draw(img_pil)
    #
    # for pt in zip(*loc[::-1]):  # Switch collumns and rows
    #     #draw.rectangle((pt[0], pt[1], pt[0] + w, pt[1] + h), outline ="red")
    #     qrcode = img_pil.crop((pt[0], pt[1], pt[0] + w, pt[1] + h))
    #     #cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), thickness=1)
    #
    # qrcode.save(outfile)
    # #cv2.imwrite(outfile, img)


def find_qr_to_file(filepath,  templatepath='0.png',  outfile="result.png",  threshold=.8):
    img = cv2.imread(filepath)
    # grayscale
    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # adaptive threshold
    #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)
    thresh0, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    #cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #res = cv2.matchTemplate(thresh, tmpl, cv2.TM_SQDIFF, mask=mask)

    # # Fill rectangular contours
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)
    #
    # # Morph open
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    #
    # # Draw rectangles, the 'area_treshold' value was determined empirically
    # cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # area_treshold = 2000
    # for c in cnts:
    #     if cv2.contourArea(c) > area_treshold:
    #         x, y, w, h = cv2.boundingRect(c)

    img_pil = Image.fromarray(thresh)
    img_pil.save(outfile)


def find_qr_to_file_my(filepath,  templatepath='11.png',  outfile="result.png",  threshold=0.6):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh0, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    template = cv2.imread(templatepath)

    w, h = template.shape[:-1]
    mask = cv2.bitwise_not(mask)
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(img,  img,  mask=mask)
    img = cv2.bitwise_not(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh1, img = cv2.threshold(gray, 0, 255, 0)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours detected:", len(contours))

    needcontours = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = float(w)/h
            if ratio >= 0.9 and ratio <= 1.1:
                needcontours.append(cnt)
    contours = []
    for cnt in needcontours:
        for cnt2 in needcontours:
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, w1, h1 = cv2.boundingRect(cnt2)
            r1 = w/w1 / 6 * 8
            r2 = h/h1 / 6 * 8
            if r1 >= 0.9 and r1 <= 1.1 and r2 >= 0.9 and r2 <= 1.1:
                contours.append(cnt2)

    print("Number of contours detected:", len(contours))

    x, y, w, h = cv2.boundingRect(contours[0])
    #cv2.drawContours(img, contours, -1, (0, 255, 255), 10)
    img_pil = Image.fromarray(img)
    img_pil = img_pil.crop((x, y, x + w, y + h))
    img_pil.save(outfile)

    #cnt = contours[0]
    #approx = cv2.approxPolyDP(cnt, epsilon, True)

    # img_pil = Image.fromarray(img)
    # img_pil.save(outfile)
    #mask = np.zeros(template.shape[:2], dtype="uint8")
    #cv2.rectangle(mask, (100, 100), (400, 400), 255, -1)

    #mask = cv2.bitwise_not(mask)
    # img_pil = Image.fromarray(img)
    # img_pil.save(outfile)


    # res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)#, mask = mask)
    #
    # #res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    # loc = np.where(res >= threshold)
    #
    # img_pil = Image.fromarray(img)
    # #draw = ImageDraw.Draw(img_pil)
    #
    # for pt in zip(*loc[::-1]):  # Switch collumns and rows
    #     #draw.rectangle((pt[0], pt[1], pt[0] + w, pt[1] + h), outline ="red")
    #     qrcode = img_pil.crop((pt[0], pt[1], pt[0] + w, pt[1] + h))
    #     #cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), thickness=1)
    #
    # qrcode.save(outfile)
    # #cv2.imwrite(outfile, img)




if __name__ == '__main__':
    find_qr_to_file_my("input.png")
    #test_encoding()
