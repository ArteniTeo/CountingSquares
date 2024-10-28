import cv2
import numpy as np

image = cv2.imread('squares.png')
if image is None:
    raise ValueError("Image not found or unable to load.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

square_count = 0

for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        side_lengths = [np.linalg.norm(approx[i] - approx[(i + 1) % 4]) for i in range(4)]
        if max(side_lengths) - min(side_lengths) < 10:  # Adjust tolerance as needed
            square_count += 1
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)

print(f"Number of squares: {square_count/4}")
cv2.imshow("Squares", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
