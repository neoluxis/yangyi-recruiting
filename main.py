import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import time

# if you want to run aotumatically,
# set `auto` to True
auto = False

def T1():
    def t1():
        img_rgb = cv.imread("task11.png")
        # cv.imshow("img_rgb", img_rgb)
        img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
        img_bin = cv.inRange(img_hsv, np.array([0, 0, 0]), np.array([100, 150, 200]))
        return [img_rgb, img_gray, img_hsv, img_bin]

    def t2(imagearr=None):
        if imagearr is None:
            imagearr = []

        def showimages(images, title=None, num_cols=2, axis='off', scale=3):
            num_images = len(images)
            num_rows = (num_images + num_cols - 1) // num_cols

            plt.figure(figsize=(num_cols * scale, num_rows * scale))

            for i, image in enumerate(images):
                plt.subplot(num_rows, num_cols, i + 1)
                plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                plt.axis(axis)
                plt.title(titles[i])

            plt.tight_layout()
            plt.show()

        titles = ['img', 'grayimg', 'hsvimg', 'binimg']
        showimages(imagearr, titles, num_cols=2, axis='off')
        time.sleep(1)
        showimages(imagearr, titles, num_cols=4, axis='on')

    t2(t1())
    time.sleep(1)
    return


def T2():
    def t1():
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        topaste = cv.imread('task11.png')
        scale = min(512 / topaste.shape[0], 512 / topaste.shape[1])
        topaste = cv.resize(topaste, (int(topaste.shape[1] * scale), int(topaste.shape[0] * scale)))
        tgt_x = (512 - topaste.shape[1]) // 2
        tgt_y = (512 - topaste.shape[0]) // 2
        img[tgt_y:tgt_y + topaste.shape[0], tgt_x:tgt_x + topaste.shape[1]] = topaste
        cv.imshow("as", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    t1()
    return


def T3():
    def judge(rgb):
        r = rgb[2]
        rgb[2] = rgb[0]
        rgb[0] = r
        if rgb[0] > 150 and rgb[1] < 30 and rgb[2] < 30:
            return 1  # red
        elif 160 < rgb[0] < 180 and 65 < rgb[1] < 90 and rgb[2] < 20:
            return 2  # yellow
        elif rgb[0] < 20 and 120< rgb[1] < 140 and 50< rgb[2] < 70:
            return 3  # green
        else:
            return 0

    vdin = cv.VideoCapture("test.mp4")
    if not vdin.isOpened():
        print("Video Not Opened")
        return
    redText = (50, 50)
    yellowTxt = (150, 50)
    greenTxt = (300, 50)
    while True:
        ret, frame = vdin.read()
        if not ret:
            break
        frame = cv.resize(frame, None, fx=0.75, fy=0.75)

        ROI = (170, 320, 100, 320)
        # cv.rectangle(frame, (ROI[0], ROI[1]), (ROI[0] + ROI[2], ROI[1] + ROI[3]), (0, 255, 0), 2)
        roi = frame[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]]
        gray = cv.cvtColor(roi, cv.COLOR_RGB2GRAY)
        _, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        filtered_contours = [contour for contour in contours if cv.contourArea(contour) >= 7]

        for contour in filtered_contours:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"]) + ROI[0]
                cY = int(M["m01"] / M["m00"]) + ROI[2] + ROI[3] * 7 // 10

                target = (cX + 20, cY + 20)
                color = frame[target[1], target[0]]
                # print(color)
                clr = judge(color)
                if clr == 1:
                    cv.putText(frame, "Red", redText, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif clr == 2:
                    cv.putText(frame, "yellow", yellowTxt, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                elif clr == 3:
                    cv.putText(frame, "green", yellowTxt, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        cv.imshow("Image with Centers", frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    vdin.release()
    cv.destroyAllWindows()


def main(auto=False):
    thu = 0
    functions = [T1, T2, T3]
    if auto:
        for f in functions:
            f()
            time.sleep(2)
    else:
        while thu < 1 or thu > 3:
            thu = int(input("Input Task: "))
        functions[thu - 1]()

def deb():
    T3()


if __name__ == "__main__":
    main(auto)
