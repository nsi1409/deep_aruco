import cv2
import onnxruntime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


#path = 'imgs/DSCF0022.JPG'
path = 'verify/DSCF0030.JPG'
print(path)

img = cv2.imread(path)
print(f'size: {img.shape}')
img = cv2.resize(img, (224, 224))
print(f'size: {img.shape}')
img = np.transpose(img, (2, 0, 1))
img = img.astype('f4')
img /= 255
img = np.expand_dims(img, axis=0)

ort_session = onnxruntime.InferenceSession('deep_aruco.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
ort_inputs = {ort_session.get_inputs()[0].name: img}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)


