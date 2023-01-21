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


def gen(data=0, outfile="out.png", size=100, dim = 8):
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
    data = ((int('1100', 2) & data) << (dim*3+1)) | ((int('0011', 2) & data ) << (dim*4+3))
    rep = '11111111'
    rep += '10000001'
    rep += '10111101'

    rep += '10100101'
    rep += '10100101'

    rep += '10111101'
    rep += '10000001'
    rep += '11111111'
    array = (bin(data | int(rep, 2))).replace("0b", "")
    #array = (bin(data)).replace("0b", "")
    matrix = [array[i*dim:(i+1)*dim] for i in range(dim)]
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


def find_qr_to_file(filepath,  templatepath='0.png',  outfile="result2.png",  threshold=.5):
    img = cv2.imread(filepath)
    #img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    #(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #thresh = 127
    #img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

    template = cv2.imread(templatepath)

    w, h = template.shape[:-1]
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
    loc = np.where(res >= threshold)

    img_pil = Image.fromarray(img)

    for pt in zip(*loc[::-1]):  # Switch collumns and rows
        qrcode = img_pil.crop((pt[0], pt[1], pt[0] + w, pt[1] + h))
        if qrcode:
            qrcode.save(outfile)

if __name__ == '__main__':
    #find_qr_to_file("input.png")
    test_encoding()
