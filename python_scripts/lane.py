import cv2
import numpy as np


def detect_lanes(image):
    """
    Detects yellow lane markers in an image and draws green edges around them.

    Args:
        image: BGR image (numpy array, as loaded by cv2.imread)

    Returns:
        output: image with green contours drawn around detected lane markers
    """
    output = image.copy()

    # Tuned to keep small lane pieces while still suppressing isolated noise.
    min_area = 20
    cleanup_kernel = np.ones((3, 3), np.uint8)

    # --- Step 1: Convert to HSV ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- Step 2: Threshold for yellow ---
    # Yellow in HSV is roughly H:15-35, S:80-255, V:80-255
    # Adjusted according to image conditions and sample points from sample_lane_hsv.py
    lower_yellow = np.array([20, 48, 80])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # --- Step 3: Morphological cleanup ---
    # Using a smaller kernel so tiny lane markers do not get erased.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cleanup_kernel)
    # CLOSE fills holes inside the markers
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cleanup_kernel)

    # --- Step 4: Find contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Step 5: Filter and draw green edges ---
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Skip tiny detections (noise)
        if area < min_area:
            continue

        # Use minAreaRect to get a tight rotated rectangle — matches the
        # expected output style with clean box outlines around each marker
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        cv2.drawContours(output, [box], 0, (0, 255, 0), 3)

    return output


def main():

    image_path = "resource/lane.png"

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not load image")
        return

    result = detect_lanes(image)

    # Show side-by-side for easy comparison
    combined = np.hstack([image, result])
    #cv2.imshow("Original  |  Lane Detection", combined)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Saving the result
    cv2.imwrite("lane_result.jpg", result)
    print("Result saved to lane_result.jpg")


if __name__ == "__main__":
    main()