import numpy as np
import cv2
import torch
from Train_pilot_v1 import AutoPilotCNN

cnn = torch.load('Autopilot_V1.pk1')

def cnn_predict(cnn, image):
    processed = pre_process_image(image)
    steering_angle = float(cnn(processed)[0].item())
    steering_angle = steering_angle * 100
    return steering_angle


def pre_process_image(img):
    image_x = 40
    image_y = 40
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    img = torch.tensor(img, dtype=torch.float32).permute(0,3,1,2)
    return img


steer = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0

cap = cv2.VideoCapture('../run.mp4')
while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
    steering_angle = cnn_predict(cnn, gray)
    print(steering_angle)
    cv2.imshow('frame', cv2.resize(frame, (500, 300), interpolation=cv2.INTER_AREA))
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
        steering_angle - smoothed_angle) / abs(
        steering_angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
