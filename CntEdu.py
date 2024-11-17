import cv2
import numpy as np


def is_square(contour, epsilon=0.05):
    # Approximate the contour shape
    approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)

    # A square will have 4 vertices
    if len(approx) == 4:
        # Check if the approximated contour forms a convex shape
        if cv2.isContourConvex(approx):
            # Calculate the lengths of each side and ensure they are approximately equal
            sides = [np.linalg.norm(approx[i] - approx[(i + 1) % 4]) for i in range(4)]
            max_side = max(sides)
            min_side = min(sides)
            if max_side - min_side < 0.1 * max_side:  # 10% tolerance on side lengths
                return True
    return False


def count_squares(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    square_count = 0

    # Loop through contours to find squares
    for contour in contours:
        if is_square(contour):
            square_count += 1
            # Draw the square on the original image
            cv2.drawContours(image, [contour], 0, (0, 255, 0), 3)

    # Display the result
    print(f"Number of squares detected: {square_count}")
    cv2.imshow("Detected Squares", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image_path = 'squares3.png'
count_squares(image_path)
