import numpy as np
import cv2
import matplotlib.pyplot as plt


# Load the image
stock_img = cv2.imread("./sotck-roadimage.png")

height, width, _ = stock_img.shape

src = np.float32([[0, height], [width, height], [0, 0], [width, 0]])
dst = np.float32(
    [[width // 2 - 50, height], [width // 2 + 50, height], [0, 0], [width, 0]]
)
H = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
print(H)
# rot_mat = cv2.Rodrigues(np.asarray([np.pi, 0, 0]))[0]
# camera_intrinsics = H @ np.linalg.inv(rot_mat)
# print(camera_intrinsics)

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

rot_mat = np.linalg.inv(camera_intrinsics) @ H
print(rot_mat)

stock_img = stock_img[height // 2 + 50 : height, 0:width]
# plt.imshow(cv2.cvtColor(stock_img, cv2.COLOR_BGR2RGB))  # Show results
# plt.show()
warped_img = cv2.warpPerspective(stock_img, H, (width, height))  # Image warping
plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))  # Show results
plt.show()
# cv2.imwrite("Warped_Image.png", warped_img)
