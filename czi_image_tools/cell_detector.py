"""
cell_detector.py
Many functions to detect cells in an image
"""
from collections import Counter
from itertools import combinations
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def preprocess_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses an image by converting it to grayscale, applying Gaussian blur,  # noqa: E501
    and thresholding.

    Parameters:
    - image_path: Path to the image file.

    Returns:
    - Tuple containing the original image and the thresholded image.
    """
    # Reading the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Converting to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying GaussianBlur to remove noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Applying thresholding to highlight the boxes
    _, thresholded_image = cv2.threshold(
        blurred_image, 200, 255, cv2.THRESH_BINARY_INV)

    return image, thresholded_image


def draw_intersection_points(image: np.ndarray, lines: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:  # noqa: E501
    """
    Draws intersection points of given lines on the image.

    Parameters:
    - image: The original image.
    - lines: Array of lines in Hough space (rho, theta).

    Returns:
    - Tuple containing the image with drawn intersections and a list of intersection points.  # noqa: E501
    """
    """Draws the intersection points of the given lines on the image."""
    intersection_points = []
    for line1, line2 in combinations(lines[:, 0], 2):
        point = find_intersection(line1, line2)
        if point:
            intersection_points.append(point)

    image_with_intersections = np.copy(image)
    for point in intersection_points:
        cv2.circle(image_with_intersections, point, 5, (0, 255, 0), -1)

    return image_with_intersections, intersection_points


def find_intersection(line1: Tuple[float, float], line2: Tuple[float, float]) -> Optional[Tuple[int, int]]:  # noqa: E501
    """
    Finds the intersection point of two lines given in Hough space in polar coordinates.  # noqa: E501

    Parameters:
    - line1: First line in Hough space (rho, theta).
    - line2: Second line in Hough space (rho, theta).

    Returns:
    - The intersection point as a tuple (x, y) or None if there is no intersection.  # noqa: E501
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([rho1, rho2])
    if np.linalg.matrix_rank(A) == 2:
        point = np.linalg.solve(A, b)
        return tuple(map(int, point))
    else:
        return None


def detect_contours(image: np.ndarray) -> List[np.ndarray]:
    """
    Detects contours in an image based on area thresholding.

    Parameters:
    - image: The thresholded image.

    Returns:
    - A list of filtered contours based on area.
    """
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 1000
    filtered_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    return filtered_contours


def draw_detected_lines(image: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """
    Draws detected lines on the image.

    Parameters:
    - image: The original image.
    - lines: Array of lines in Hough space (rho, theta).

    Returns:
    - The image with detected lines drawn on it.
    """
    h, w, _ = image.shape
    diagonal = int(np.sqrt(h**2 + w**2))
    line_image = np.copy(image)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + diagonal * (-b))
            y1 = int(y0 + diagonal * (a))
            x2 = int(x0 - diagonal * (-b))
            y2 = int(y0 - diagonal * (a))
            cv2.line(img=line_image, pt1=(x1, y1), pt2=(
                x2, y2), color=(0, 0, 255), thickness=2)
    return line_image


def apply_hough_transform(edges: np.ndarray, threshold: int = 800) -> np.ndarray:  # noqa: E501
    """
    Applies the Hough transform to detect lines in an edge image.

    Parameters:
    - edges: The edge-detected image.
    - threshold: Threshold for the Hough transform.

    Returns:
    - Array of detected lines in Hough space (rho, theta).
    """
    lines = cv2.HoughLines(
        image=edges, rho=1, theta=np.pi / 180, threshold=threshold)
    return lines


def enhance_lines(edges: np.ndarray, kernel_vertical: np.ndarray, kernel_horizontal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: E501
    """
    Enhances vertical and horizontal lines in an edge-detected image.

    Parameters:
    - edges: The edge-detected image.
    - kernel_vertical: Kernel for enhancing vertical lines.
    - kernel_horizontal: Kernel for enhancing horizontal lines.

    Returns:
    - Tuple containing enhanced vertical lines, horizontal lines, and combined lines.  # noqa: E501
    """
    vertical_lines = cv2.dilate(
        src=edges, kernel=kernel_vertical, iterations=2)
    vertical_lines = cv2.erode(
        src=vertical_lines, kernel=kernel_vertical, iterations=1)

    horizontal_lines = cv2.dilate(
        src=edges, kernel=kernel_horizontal, iterations=2)
    horizontal_lines = cv2.erode(
        src=horizontal_lines, kernel=kernel_horizontal, iterations=1)

    combined_lines = cv2.addWeighted(
        src1=vertical_lines, alpha=0.5, src2=horizontal_lines, beta=0.5, gamma=0.0)  # noqa: E501

    return vertical_lines, horizontal_lines, combined_lines


def apply_canny_edge_detection(image: np.ndarray) -> np.ndarray:
    """
    Applies Canny edge detection to an image.

    Parameters:
    - image: The original image.

    Returns:
    - The edge-detected image.
    """
    gray_roi = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(src=gray_roi, ksize=(5, 5), sigmaX=0)
    edges = cv2.Canny(image=blurred_roi, threshold1=50, threshold2=150)
    return edges


def display_image(img: np.ndarray, title: str = "") -> None:
    """
    Displays an image using Matplotlib.

    Parameters:
    - img: The image to display.
    - title: The title of the image window.
    """
    plt.imshow(X=cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB))
    plt.title(label=title)
    plt.axis(False)
    plt.show()


def extract_main_content(img_path: str) -> Optional[np.ndarray]:
    """
    Extracts the main content from an image based on the largest contour area.

    Parameters:
    - img_path: Path to the image file.

    Returns:
    - The largest subimage extracted from the original image or None if no contour is found.  # noqa: E501
    """
    original_image, processed_image = preprocess_image(image_path=img_path)
    contours = detect_contours(image=processed_image)

    largest_contour = None
    largest_area = 0

    for cnt in contours:
        area = cv2.contourArea(contour=cnt)

        if area > largest_area:
            largest_area = area
            largest_contour = cnt

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(array=largest_contour)
        largest_subimage = original_image[y:y+h, x:x+w]
        return largest_subimage
    else:
        return None


def filter_close_lines(lines: np.ndarray, threshold: int = 20) -> Tuple[np.ndarray, int]:  # noqa: E501
    """
    Filters lines that are too close to each other based on a threshold.

    Parameters:
    - lines: Detected lines in Hough space (rho, theta).
    - threshold: Distance threshold for filtering.

    Returns:
    - Tuple containing the filtered lines in Hough space and the number of filtered lines.  # noqa: E501
    """
    if lines is None:
        return [], 0

    sorted_lines = sorted(lines[:, 0], key=lambda x: x[0])

    filtered_lines = []
    prev_line = sorted_lines[0]
    filtered_lines.append(prev_line)

    for curr_line in sorted_lines[1:]:
        rho_diff = abs(curr_line[0] - prev_line[0])
        theta_diff = abs(curr_line[1] - prev_line[1])

        if rho_diff > threshold or theta_diff > np.pi / 180:
            filtered_lines.append(curr_line)
            prev_line = curr_line

    filtered_lines = np.array(filtered_lines).reshape(-1, 1, 2)
    num_filtered_lines = len(filtered_lines)

    return filtered_lines, num_filtered_lines


def filter_lines(lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters horizontal and vertical lines from the detected lines.

    Parameters:
    - lines: Detected lines in Hough space (rho, theta).

    Returns:
    - Tuple containing filtered vertical and horizontal lines.
    """
    horizontal_lines = np.array(
        [line for line in lines[:, 0] if 0.7 < line[1] < 2.5]).reshape(-1, 1, 2)  # noqa: E501
    vertical_lines = np.array(
        [line for line in lines[:, 0] if line[1] < 0.4 or line[1] > 2.8]).reshape(-1, 1, 2)  # noqa: E501

    filtered_vertical_lines, num_filtered_vertical_lines = filter_close_lines(
        vertical_lines)
    filtered_horizontal_lines, num_filtered_horizontal_lines = filter_close_lines(  # noqa: E501
        horizontal_lines)

#     filtered_vertical_line_image = draw_detected_lines(largest_subimage, filtered_vertical_lines)  # noqa: E501
#     filtered_horizontal_line_image = draw_detected_lines(largest_subimage, filtered_horizontal_lines)  # noqa: E501
    return filtered_vertical_lines, filtered_horizontal_lines


def contains_cells(image: np.ndarray, display: bool = False, threshold_ratio=0.01) -> bool:  # noqa: E501
    """
    Checks if an image contains cells based on color thresholding.

    Parameters:
    - image: The original image.
    - display: Whether to display the masked image.
    - threshold_ratio: The threshold ratio of masked area to the total area.

    Returns:
    - True if cells are present, False otherwise.
    """
    hsv = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)

    lower_bound = np.array(object=[125, 50, 50])
    upper_bound = np.array(object=[170, 255, 255])

    mask = cv2.inRange(src=hsv, lowerb=lower_bound, upperb=upper_bound)

    area = mask.shape[0] * mask.shape[1]
    masked_area = np.sum(a=mask == 255)
    ratio = masked_area / area

    if display:
        if ratio > threshold_ratio:
            display_image(img=mask, title="Masked - Contains Cells")
        else:
            display_image(img=mask, title="Masked - Doesn't Contain Cells")

    return ratio > threshold_ratio


def edge_detection_pipeline(img_path: str, display: bool = False) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:  # noqa: E501
    """
    Processes an image to detect edges, lines, and intersection points.

    Parameters:
    - img_path: Path to the image file.
    - display: Whether to display intermediate results.

    Returns:
    - Tuple containing the largest subimage, detected lines, and intersection points.  # noqa: E501
    """
    largest_subimage = extract_main_content(img_path=img_path)
    edges_detected = apply_canny_edge_detection(image=largest_subimage)

    kernel_vertical = np.ones(shape=(5, 1), dtype=np.uint8)
    kernel_horizontal = np.ones(shape=(1, 5), dtype=np.uint8)
    vertical_lines, horizontal_lines, combined_lines = enhance_lines(
        edges=edges_detected, kernel_vertical=kernel_vertical, kernel_horizontal=kernel_horizontal)  # noqa: E501
    lines = apply_hough_transform(edges=combined_lines, threshold=800)
    line_image = draw_detected_lines(image=largest_subimage, lines=lines)
    image_with_intersections, intersection_points = draw_intersection_points(
        image=largest_subimage, lines=lines)
    if display:
        display_image(
            img=line_image, title="Lines Detected (Refactored with Threshold 800)")  # noqa: E501
        display_image(img=image_with_intersections,
                      title="Intersection Points (Refactored with Threshold 800)")  # noqa: E501
    return largest_subimage, lines, intersection_points


def most_repeated_count_and_segments(segmented_images: List[np.ndarray]) -> Tuple[int, List[np.ndarray]]:  # noqa: E501
    """
    Finds the most repeated cell count among different detection methods and the corresponding segments.  # noqa: E501

    Parameters:
    - segmented_images: List of segmented images.

    Returns:
    - Tuple containing the most common count and the list of segments corresponding to that count.  # noqa: E501
    """
    rounded_shapes_count = 0
    colored_pixels_count = 0
    both_approaches_count = 0

    rounded_shapes_segments = []
    colored_pixels_segments = []
    both_approaches_segments = []

    for segment in segmented_images:
        if segment.size == 0:
            continue

        has_rounded_shapes = detect_rounded_shapes(image=segment.copy())
        has_colored_pixels = detect_colored_pixels(image=segment.copy())

        if has_rounded_shapes:
            rounded_shapes_count += 1
            rounded_shapes_segments.append(segment)

        if has_colored_pixels:
            colored_pixels_count += 1
            colored_pixels_segments.append(segment)

        if has_rounded_shapes and has_colored_pixels:
            both_approaches_count += 1
            both_approaches_segments.append(segment)

    counts = [rounded_shapes_count,
              colored_pixels_count, both_approaches_count]
    count_frequency = Counter(counts)
    most_common_count, _ = count_frequency.most_common(1)[0]

    most_common_method_segments = []
    if most_common_count == rounded_shapes_count:
        most_common_method_segments = rounded_shapes_segments
    elif most_common_count == colored_pixels_count:
        most_common_method_segments = colored_pixels_segments
    elif most_common_count == both_approaches_count:
        most_common_method_segments = both_approaches_segments

    return most_common_count, most_common_method_segments


def detect_rounded_shapes(image: np.ndarray, display: bool = False) -> bool:
    """
    Detects rounded shapes (cells) in an image.

    Parameters:
    - image: The image to process.
    - display: Whether to display the processed image.

    Returns:
    - True if rounded shapes are detected, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) >= 8:
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 5)

    if display:
        display_image(image, "Detected Rounded Shapes")

    return len(contours) > 0


def detect_colored_pixels(image: np.ndarray, display: bool = False) -> bool:
    """
    Detects colored pixels in an image that represent cells.

    Parameters:
    - image: The image to process.
    - display: Whether to display the processed image.

    Returns:
    - True if colored pixels are detected, False otherwise.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([125, 50, 50])
    upper_bound = np.array([170, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    detected_area = np.sum(mask == 255)

    if display:
        display_image(mask, "Detected Colored Pixels")

    return detected_area > 500  # Threshold can be adjusted


def proportional_crop(image: np.ndarray, crop_percentage: float = 0.84, offset_percentage: float = 0.08) -> np.ndarray:  # noqa: E501
    """
    Crop the image proportionally based on given percentages.

    Parameters:
    - image: The original image to be cropped.
    - crop_percentage: The percentage of the original image to keep after cropping.  # noqa: E501
    - offset_percentage: The percentage to offset the crop from the edge of the original image.  # noqa: E501

    Returns:
    - cropped_image: The cropped image.
    """
    original_height, original_width, _ = image.shape

    crop_height = int(original_height * crop_percentage)
    crop_width = int(original_width * crop_percentage)

    offset_height = int(original_height * offset_percentage)
    offset_width = int(original_width * offset_percentage)

    cropped_image = image[offset_height:offset_height +
                          crop_height, offset_width:offset_width + crop_width, :]  # noqa: E501

    return cropped_image


def segment_image(image_path, display=False):
    """
    Segment the image using the detected vertical and horizontal lines.

    Parameters:
    - image: The image to be segmented.
    - vertical_lines: The detected vertical lines in Hough space (rho, theta).
    - horizontal_lines: The detected horizontal lines in Hough space (rho, theta).  # noqa: E501

    Returns:
    - segments: A list of segmented sub-images.
    """
    segments = []

    image, lines, intersection_points = edge_detection_pipeline(
        image_path, display=display)
    vertical_lines, horizontal_lines = filter_lines(lines)

    vertical_lines = sorted(vertical_lines[:, 0], key=lambda x: x[0])
    horizontal_lines = sorted(horizontal_lines[:, 0], key=lambda x: x[0])

    vertical_cuts = [0]
    horizontal_cuts = [0]

    h, w, _ = image.shape
    for rho, theta in vertical_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        vertical_cuts.append(int(x0))
    for rho, theta in horizontal_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        y0 = b * rho
        horizontal_cuts.append(int(y0))

    vertical_cuts.append(w)
    horizontal_cuts.append(h)

    for i in range(len(vertical_cuts) - 1):
        for j in range(len(horizontal_cuts) - 1):
            x1, x2 = vertical_cuts[i], vertical_cuts[i + 1]
            y1, y2 = horizontal_cuts[j], horizontal_cuts[j + 1]
            segment = image[y1:y2, x1:x2]
            segments.append(segment)
    most_common_count, common_segments = most_repeated_count_and_segments(
        segments)
    sample_cropped_images = [proportional_crop(img) for img in common_segments]
    return sample_cropped_images


if __name__ == "__main__":
    import os
    image_path = "images_output/2019_12_09__09_49__0001/2019_12_09__09_49__0001"
    files = os.listdir(image_path)
    image = cv2.imread(os.path.join(image_path, files[0]))
    contains_cells(image=image, display=True)
