import cv2
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import trochmodle

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
            img = np.array(gray) / 255
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img)
            img = torch.tensor(img)
            img = img.to(torch.float32)
            newmoo = trochmodle.HandsNetModel()
            retC = torch.load('model_01.pt')
            newmoo.load_state_dict(retC['net'])

            output = newmoo(img)
            ind,pred = torch.max(output.data, dim=1)
            print("结果", pred+1)
            if cv2.waitKey(1) == ord('L'):

                break
cap.release()
cv2.destroyAllWindows()
