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

stock_img = stock_img[
    height // 2 + 50 : height, 0:width
]  # Apply np slicing for ROI crop
# plt.imshow(cv2.cvtColor(stock_img, cv2.COLOR_BGR2RGB))  # Show results
# plt.show()
warped_img = cv2.warpPerspective(stock_img, H, (width, height))  # Image warping
# plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))  # Show results
# plt.show()
cv2.imwrite("Warped_Image.png", warped_img)
