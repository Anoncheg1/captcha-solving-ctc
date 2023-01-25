import os
import shutil

import cv2

import cv2 as cv

import numpy as np

from utils.grayscale import show_two_images, clear_captcha

# 6 и б отличаются
# т и г тяжело отличить
# п и н тяжело отличить
# нету e
# Press the green button in the gutter to run the script.


# https://bestofphp.com/repo/Gregwar-Captcha-php-image-processing

def to_gray_random(p1, p2, howm):
    import random
    shutil.rmtree(p2)
    os.mkdir(p2)
    ld = list(os.listdir(p1))
    for i in range(howm):
        ldi = random.randint(0, len(ld)-1)
        print(i, ldi)
        filename = ld[ldi]
        # solv = filename[0:5]
        file = os.path.join(p1, filename)
        # inv = [4, 8, 67, 72, 80, 95]

        img: np.ndarray = cv.imread(file)

        gray = clear_captcha(img)
        cv.imwrite(os.path.join(p2, filename), gray)


def to_gray(p1, p2):
    shutil.rmtree(p2)
    os.mkdir(p2)
    ld = list(os.listdir(p1))
    print("(len(ld))", len(ld))
    for i in range(len(ld)):
        # ldi = random.randint(0, len(ld)-1)
        # print(i)
        filename = ld[i]

        # solv = filename[0:5]
        file = os.path.join(p1, filename)

        # inv = [4, 8, 67, 72, 80, 95]

        img: np.ndarray = cv.imread(file)
        # cv.imshow('image', img)  # show image in window
        # cv.waitKey(0)  # wait for any key indefinitely
        # cv.destroyAllWindows()  # close window q
        gray = clear_captcha(img)

        filename = filename.lower()
        if '-' in filename:
            filename = filename.split('-')[0] + '.jpg'
        if 'r' in filename:
            filename = ''.join(['г' if x == 'r' else x for x in filename])

        cv.imwrite(os.path.join(p2, filename), gray)


if __name__ == '__main__':

    p1 = './phptest/gen_train/'
    p2 = './train/'

    # to_gray_random(p1, p2, 40000)

    p1 = 'jpg1/'
    p2 = 'test/'

    # to_gray(p1, p2)  # 408

    p1 = 'jpg2/'
    p2 = 'jpg2_gray/'

    # to_gray(p1, p2)  # 1162

    # for filename in os.listdir(p2):
    #     file = os.path.join(p1, filename)
    #     file2 = os.path.join('./test/', filename)
    #     shutil.copy(file, file2)
    p1 = '09_01_23_alpha/'
    p2 = 'train/'
    to_gray(p1, p2)

