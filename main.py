import re
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def parse(istr):
    pattern = r'^(\d)\.(\d)$'
    match = re.search(pattern, istr)

    if match:
        left_side = int(match.group(1))
        right_side = int(match.group(2))
        if 0 < left_side <= 3 and 0 < right_side <= 3:
            return True, left_side, right_side
    return False, 1, 1


def T1(s=1):
    def t1():
        img_rgb = cv.imread("task11.png")
        # cv.imshow("img_rgb", img_rgb)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
        img_bin = cv.inRange(img_hsv, np.array([0, 0, 0]), np.array([100, 150, 200]))
        cv.imshow("img", img_rgb)
        cv.waitKey(0)
        cv.imshow("img", img_gray)
        cv.waitKey(0)
        cv.imshow("img", img_hsv)
        cv.waitKey(0)
        cv.imshow("img", img_bin)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return [img_rgb, img_gray, img_hsv, img_bin]

    def t2(imagearr=[]):
        def showimages(images, title=None, num_cols=2, axis='off', scale=3):
            """
            一个窗口显示多张图片
            :param images:
            :param title:
            :param num_cols:
            :param axis:
            :param scale:
            :return:
            """
            num_images = len(images)
            num_rows = (num_images + num_cols - 1) // num_cols

            plt.figure(figsize=(num_cols * scale, num_rows * scale))

            for i, image in enumerate(images):
                plt.subplot(num_rows, num_cols, i + 1)
                plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))  # 将BGR格式转换为RGB以正确显示图像
                plt.axis(axis)
                plt.title(titles[i])

            plt.tight_layout()
            plt.show()

        titles = ['img', 'grayimg', 'hsvimg', 'binimg']
        showimages(imagearr, titles, num_cols=2, axis='on')

    functions = [None, t1, t2]

    # functions[s]()
    t2(t1())

def main():
    r = False
    thu, tsm = 1, 1
    while not r:
        r, thu, tsm = parse(input("Input Task: "))


def deb():
    T1(2)


if __name__ == "__main__":
    deb()
