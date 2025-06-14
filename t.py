# import cudaq
# print(cudaq.__version__)  # cudaq 버전 확인

import torch

print(torch.cuda.is_available())  # GPU 사용 가능 여부 확인
print(torch.version.cuda)         # CUDA 버전 확인

# import cv2
# import numpy as np

# print(cv2.__version__)  # OpenCV 버전 확인
# print(np.__version__)   # numpy 버전 확인