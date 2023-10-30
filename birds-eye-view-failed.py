import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the image
stock_img = cv2.imread("./sotck-roadimage.png")
height, width, _ = stock_img.shape
stock_img = stock_img[height // 2 + 50 : height, 0:width]

# ref: https://nikolasent.github.io/opencv/2017/05/07/Bird's-Eye-View-Transformation.html

camera_intrinsics = np.array(
    [
        500.0,
        0.000000,
        width // 2,
        0.000000,
        500.0,
        height // 2,
        0.000000,
        0.000000,
        1.000000,
    ]
).reshape(3, 3)

rot_mat = np.array(
    [
        2.00000000e-03,
        -2.63852243e-06,
        -5.46000000e-01,
        0.00000000e00,
        6.48179420e-03,
        -3.78000000e-01,
        0.00000000e00,
        1.17941953e-02,
        1.00000000e00,
    ]
).reshape(3, 3)

# print(rot_mat)
# rot_mat = cv2.Rodrigues(np.array([np.pi / 3.0, 0.0, 0.0]))[0]

# rot_mat[:, 2] = np.array([-5.46000000e-01, -3.78000000e-01, 1.0])
# print(rot_mat)

# print(rot_mat)
# range_ = np.arange(-100, 100)
# #
# min_rot_mat = np.zeros((3, 3))
# fin = []
# min = np.inf
# # # # for i in range_:
# # # #     if i == 0:
# # # #         continue
# for i in tqdm(range_):
#     if i == 0:
#         continue
#     for j in range_:
#         if j == 0:
#             continue
#         for k in range_:
#             if i == 0 or j == 0 or k == 0:
#                 continue
#             # 10, -2, 5
#             test_rot_mat = cv2.Rodrigues(np.array([np.pi / i, np.pi / j, np.pi / k]))[0]
#             test_rot_mat[:, 2] = np.array([0.0, 0.0, 1.0])

#             if np.linalg.norm(test_rot_mat - rot_mat) < min:
#                 min = np.linalg.norm(test_rot_mat - rot_mat)
#                 min_rot_mat = test_rot_mat
#                 fin = [i, j, k]
# print(f"{fin=}")
# print(f"{min_rot_mat=}")
# print(f"{min=}")

H = camera_intrinsics @ rot_mat

# print(H)
# print(rot_mat)

warped_img = cv2.warpPerspective(stock_img, H, (width, height))  # Image warping

plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))  # Show results
plt.show()
