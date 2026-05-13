# distance.py

import cv2
import numpy as np
import glob

# =========================
# 1) CAMERA CALIBRATION
# =========================
CHECKERBOARD = (6, 8)          # inner corners
square_size = 0.025            # meters (25 mm)

def calibrate_camera():
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    images = glob.glob('calibration/*.png')

    image_size = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            image_size = gray.shape[::-1]

    if len(objpoints) == 0:
        raise RuntimeError("No valid checkerboard detections found")

    ret, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("Intrinsic Matrix K:\n", K)
    return K, dist


# =========================
# 2) CLICK PIXEL HELPER
# =========================
def get_pixel_from_click(image_path):
    img = cv2.imread(image_path)

    coords = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked pixel: ({x}, {y})")
            coords.append((x, y))

    cv2.imshow("Click lower-right cone corner", img)
    cv2.setMouseCallback("Click lower-right cone corner", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(coords) == 0:
        raise RuntimeError("No pixel selected")

    return coords[0]

# =========================
# 3) COMPUTE CAMERA HEIGHT
# =========================
def compute_height(K, u, v, x_car):
    fx = K[0, 0]
    fy = K[1, 1]
    x0 = K[0, 2]
    y0 = K[1, 2]
    x = (u - x0)/fx
    y = (v - y0)/fy

    height = y*x_car

    return height

# =========================
# 4) PIXEL TO CAR FRAME
# =========================
def pixel_to_car(K, u, v, H):
    fx = K[0, 0]
    fy = K[1, 1]
    x0 = K[0, 2]
    y0 = K[1, 2]

    x_car = (fy*H)/(v - y0)
    y_car = ((u - x0)/fx)*x_car

    return x_car, y_car


# =========================
# 5) MAIN PIPELINE
# =========================
def main():
    # Step 1: calibrate
    K, dist = calibrate_camera()

    # Step 2: known cone (40 cm)
    print("\nClick cone in cone_x40cm.png")
    u_known, v_known = get_pixel_from_click('resource/cone_x40cm.png')

    x_known = 0.40  # meters

    H = compute_height(K, u_known, v_known, x_known)
    print(f"\nEstimated camera height H: {H:.4f} m")

    # Step 3: unknown cone
    print("\nClick cone in cone_unknown.png")
    u_unknown, v_unknown = get_pixel_from_click('resource/cone_unknown.png')

    x, y = pixel_to_car(K, u_unknown, v_unknown, H)

    print(f"\nEstimated position of unknown cone:")
    print(f"x_car = {x:.4f} m")
    print(f"y_car = {y:.4f} m")

if __name__ == "__main__":
    main()