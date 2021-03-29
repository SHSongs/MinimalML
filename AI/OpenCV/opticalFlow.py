# calcOpticalFlowPyrLK 추적 (track_opticalLK.py)

import cv2
import numpy as np
import time

cap = cv2.VideoCapture('jump.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
print(fps)
print(delay)
color = np.random.randint(0, 255, (200, 3))

lines = None
prevImg = None


termcriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

while cap.isOpened():
    ret, frame = cap.read()

    scale_percent = 50  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    if not ret:
        break
    img_draw = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 최초 프레임 경우
    if prevImg is None:
        prevImg = gray
        # 추적선 그릴 이미지를 프레임 크기에 맞게 생성
        lines = np.zeros_like(frame)
        # 추적 시작을 위한 코너 검출  ---①
        prevPt = cv2.goodFeaturesToTrack(prevImg, 200, 0.01, 10)
    else:
        nextImg = gray

        nextPt, status, err = cv2.calcOpticalFlowPyrLK(prevImg, nextImg, \
                                                       prevPt, None, criteria=termcriteria)

        prevMv = prevPt[status == 1]
        nextMv = nextPt[status == 1]
        vec = prevMv - nextMv

        vec = np.mean(vec, axis=0)
        print(vec)

        # if vec[0] > 0:
        #     print('우')
        # elif vec[0] < 0:
        #     print('왼')
        if abs(vec[1]) > 0.3:
            if vec[1] > 0:
                print('위')
            elif vec[1] < 0:
                print('아래')
        for i, (p, n) in enumerate(zip(prevMv, nextMv)):
            px, py = p.ravel()
            nx, ny = n.ravel()

            cv2.line(lines, (px, py), (nx, ny), color[i].tolist(), 2)

            cv2.circle(img_draw, (nx, ny), 2, color[i].tolist(), -1)

        img_draw = cv2.add(img_draw, lines)

        prevImg = nextImg
        prevPt = nextMv.reshape(-1, 1, 2)

    cv2.imshow('OpticalFlow-LK', img_draw)
    key = cv2.waitKey(delay)
    if key == 27:  # Esc:
        break
    elif key == 8:  # Backspace:
        prevImg = None
    elif key == ord('a'):
        time.sleep(1)
cv2.destroyAllWindows()
cap.release()
