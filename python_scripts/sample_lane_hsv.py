import argparse

import cv2
import numpy as np


def sample_patch(image, x, y, patch_size):
    half = patch_size // 2
    y0 = max(0, y - half)
    y1 = min(image.shape[0], y + half + 1)
    x0 = max(0, x - half)
    x1 = min(image.shape[1], x + half + 1)
    patch = image[y0:y1, x0:x1]
    return patch, (x0, y0, x1, y1)


def main():
    parser = argparse.ArgumentParser(description="Click pixels on a lane image and print BGR/HSV values.")
    parser.add_argument("--image", default="resource/lane.png", help="Path to the image to inspect")
    parser.add_argument("--patch-size", type=int, default=7, help="Odd patch size to average around each click")
    args = parser.parse_args()

    if args.patch_size < 1 or args.patch_size % 2 == 0:
        raise SystemExit("patch-size must be a positive odd integer")

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not load image: {args.image}")

    display = image.copy()
    window_name = "Click a lane pixel, press q to quit"

    def on_mouse(event, x, y, flags, param):
        nonlocal display
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        patch, bounds = sample_patch(image, x, y, args.patch_size)
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        bgr_mean = patch.mean(axis=(0, 1))
        hsv_mean = hsv_patch.mean(axis=(0, 1))

        print(f"Point: ({x}, {y})")
        print(f"Patch bounds: x={bounds[0]}:{bounds[2]}, y={bounds[1]}:{bounds[3]}")
        print(f"Mean BGR: {bgr_mean.round(2)}")
        print(f"Mean HSV: {hsv_mean.round(2)}")
        print("Suggested HSV bounds around the patch mean:")
        print(f"  lower = [{max(0, int(hsv_mean[0]) - 10)}, {max(0, int(hsv_mean[1]) - 80)}, {max(0, int(hsv_mean[2]) - 80)}]")
        print(f"  upper = [{min(179, int(hsv_mean[0]) + 10)}, {min(255, int(hsv_mean[1]) + 80)}, {min(255, int(hsv_mean[2]) + 80)}]")
        print()

        cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
        cv2.imshow(window_name, display)

    cv2.imshow(window_name, display)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()