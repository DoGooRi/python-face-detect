import dlib
import cv2, sys
import numpy as np

scale = 0.7

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture('asset/video.mp4')

while True:
    ret, img = cap.read()

    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1]*scale), (int(img.shape[0]*scale))))

    # 얼굴 감지
    faces = detector(img)
    face = faces[0]

    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # 시각화 처리
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break