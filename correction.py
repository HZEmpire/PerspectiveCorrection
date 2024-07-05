import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm

# Function for find lines of the image
def find_long_ling(img):
    """
    input:
        img: image loaded by cv2.imread

    return: 
        horizontal_lines: list of potential horizontal lines in descending order of length
        vertical_lines: list of potential vertical lines in descending order of length

    """
    # Long lines detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 80, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=min(img.shape[:2])*0.1, maxLineGap=30)
    img_h, img_w = img.shape[:2]

    # Find horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if not x1 == x2 and abs((y2 - y1) / (x2 - x1)) < 0.16:  # near horizontal line
            # Skip lines that are close to the image border
            if y1 < img_h * 0.05 or y1 > img_h * 0.95 or y2 < img_h * 0.05 or y2 > img_h * 0.95:
                continue
            horizontal_lines.append(line)
        elif not y1 == y2 and abs((x2 - x1) / (y2 - y1)) < 0.16: # near vertical
            # Skip lines that are close to the image border
            if x1 < img_w * 0.05 or x1 > img_w * 0.95 or x2 < img_w * 0.05 or x2 > img_w * 0.95:
                continue
            vertical_lines.append(line)

    # Sort by length for horizontal lines, descending
    horizontal_lines.sort(key=lambda x: (x[0][0] - x[0][2])**2 + (x[0][1] - x[0][3])**2, reverse=True)

    # Sort by length for vertical lines, descending
    vertical_lines.sort(key=lambda x: (x[0][0] - x[0][2])**2 + (x[0][1] - x[0][3])**2, reverse=True)
        
    return horizontal_lines, vertical_lines

# Function for find the intersection of the lines
def get_intersection(line1, line2):
    """
    input:
        line1: list of two points of line 1
        line2: list of two points of line 2
    
    return:
        x, y: intersection point of line1 and line2
    """
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return x, y

# Function for find the four points for the rectangle
def find_rectangle(horizontal_lines, vertical_lines, height, width):
    """
    input:
        horizontal_lines: list of potential horizontal lines in descending order of length
        vertical_lines: list of potential vertical lines in descending order of length
        height: height of the original image
        width: width of the original image

    return:
        p1, p2, p3, p4: 4 points of the rectangle
    """
    # Find 2 longest horizontal lines
    # Make sure the 2 lines are at least 10% width close to each other
    line1 = horizontal_lines[0]
    line2 = None
    for line2 in horizontal_lines[1:]:
        if abs(line1[0][1] - line2[0][1]) > height * 0.1:
            break
    # Swap line1 and line2 if line1 is below line2
    if line1[0][1] > line2[0][1]:
        line1, line2 = line2, line1

    # Find 2 longest vertical lines
    # Make sure the 2 lines are at least 10% height close to each other
    line3 = vertical_lines[0]
    line4 = None
    for line4 in vertical_lines[1:]:
        if abs(line3[0][0] - line4[0][0]) > width * 0.1:
            break
    # Swap line3 and line4 if line3 is on the right of line4
    if line3[0][0] > line4[0][0]:
        line3, line4 = line4, line3

    # Get intersection point of 4 lines
    p1 = get_intersection(line1, line3)
    p2 = get_intersection(line1, line4)
    p3 = get_intersection(line2, line3)
    p4 = get_intersection(line2, line4)

    return p1, p2, p3, p4

# Function for correct the perspective of the image
def perspective_correction(img, point_list):
    """
    input:
        img: image loaded by cv2.imread
        point_list: 4 points of the rectangle

    return:
        corrected_img: image after perspective correction
    """
    # Define the points for the rectangle
    p1, p2, p3, p4 = point_list
    pts1 = np.float32([p1, p2, p3, p4])
    
    # Perspective transform height and width
    h = int((abs(p1[1]-p3[1]) + abs(p2[1]-p4[1])) / 2)
    w = int((abs(p1[0]-p2[0]) + abs(p3[0]-p4[0])) / 2)
    # Define the points for the rectangle after correction
    pts2 = np.float32([[p1[0] + 400, p1[1] + 400], [p1[0]+w + 400, p1[1]+ 400], [p1[0] + 400, p1[1]+h +400], [p1[0]+w+400, p1[1]+h+400]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # Perspective transform
    corrected_img = cv2.warpPerspective(img, M, (img.shape[1]+800, img.shape[0]+800))


    def crop_black_borders(image):
        """
        input:
            image: image after perspective correction

        return:
            corrected: image after crop the black borders
        """
        y_nonzero, x_nonzero, _ = np.nonzero(image)
        return image[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    corrected = crop_black_borders(corrected_img)
    corrected = cv2.resize(corrected, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    return corrected

def main():
    parser = argparse.ArgumentParser(description='Correct the perspective of the image')
    parser.add_argument(
        "--input",
        "-I",
        type=str,
        default="./input/",
        help="Path to the input image",
    )
    parser.add_argument(
        "--output",
        "-O",
        type=str,
        default="./output/",
        help="Path to the output image",
    )
    args = parser.parse_args()

    # Batch processing
    input_folder = args.input

    # Grab all jpg, jpeg, png files in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    print(f"Found {len(files)} files")

    # Create the output folder if it does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Process all files
    for file in tqdm(files):
        img = cv2.imread(input_folder + file, 1)
        horizontal_lines, vertical_lines = find_long_ling(img)
        intersection_points = find_rectangle(horizontal_lines, vertical_lines, img.shape[0], img.shape[1])
        corrected_img = perspective_correction(img, intersection_points)
        cv2.imwrite('output/' + file, corrected_img) 

if __name__ == "__main__":
    main()