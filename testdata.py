import torch
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('正在打开')
else:
    ir = 0
    while True:

        ret, frame = cap.read()
        if not ret:
            break
        else:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('ff', gray)
            if cv2.waitKey(1) == ord('L'):
                cv2.imwrite('dataset\\hands\\{}.jpg'.format(ir+40), gray)
                ir = ir + 1
                if ir == 10:
                    break
cap.release()
cv2.destroyAllWindows()
