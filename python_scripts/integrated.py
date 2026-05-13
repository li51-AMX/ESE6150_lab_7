#!/usr/bin/env python3

import cv2
import numpy as np
import pyrealsense2 as rs
import tensorrt as trt
import pycuda.driver as cuda
import time
import glob

from lane import detect_lanes

# =========================
# CONFIG
# =========================
TRT_ENGINE_PATH = "model/model_fp16.trt"
CONF_THRESH = 0.74
NMS_THRESH = 0.4
SAVE_PATH = "frame.jpg"
ENABLE_LANE_DETECTION = True

CAMERA_HEIGHT_H = 0.137  # meters — get this from compute_height() once
K = np.array([[693.7,  0, 448.5],
              [ 0, 694.8, 258.1],
              [ 0,  0,  1]], dtype=np.float32)

# =========================
# CUDA
# =========================
cuda.init()
cuda_ctx = cuda.Device(0).make_context()

# =========================
# UTILS
# =========================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def nms(boxes, scores, thresh=0.4):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return keep

def pixel_to_car(K, u, v, H):
    fx = K[0, 0]
    fy = K[1, 1]
    x0 = K[0, 2]
    y0 = K[1, 2]
    x_car = (fy * H) / (v - y0)
    y_car = ((u - x0) / fx) * x_car
    return x_car, y_car

# =========================
# TRT MODEL
# =========================
class TRTModel:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append((name, host_mem, device_mem))
            else:
                self.outputs.append((name, host_mem, device_mem))

        self.stream = cuda.Stream()

    def infer(self, input_data):
        name_in, host_in, device_in = self.inputs[0]
        name_out, host_out, device_out = self.outputs[0]

        np.copyto(host_in, input_data.ravel())

        cuda.memcpy_htod_async(device_in, host_in, self.stream)

        self.context.set_tensor_address(name_in, int(device_in))
        self.context.set_tensor_address(name_out, int(device_out))

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(host_out, device_out, self.stream)
        self.stream.synchronize()

        return host_out

# =========================
# DECODER
# =========================

def decode(output, input_w, input_h):
    GRID_ROWS = 5
    GRID_COLS = 10
    cell_w = input_w / GRID_COLS   # 32px
    cell_h = input_h / GRID_ROWS   # 36px

    # reshape flat output back into (5, 5, 10)
    output = output.reshape(5, GRID_ROWS, GRID_COLS)

    boxes = []
    scores = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            conf = sigmoid(output[0, row, col])

            if conf < CONF_THRESH:
                continue

            dx = sigmoid(output[1, row, col])
            dy = sigmoid(output[2, row, col])

            cx = (col + dx) * cell_w
            cy = (row + dy) * cell_h

            w = sigmoid(output[3, row, col]) * input_w
            h = sigmoid(output[4, row, col]) * input_h


            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

    return np.array(boxes), np.array(scores)

# =========================
# MAIN
# =========================
def main():
    try:
        # ---- Load TensorRT ----
        model = TRTModel(TRT_ENGINE_PATH)

        name = model.engine.get_tensor_name(0)
        _, _, input_h, input_w = model.engine.get_tensor_shape(name)

        print(f"[INFO] Model input: {input_w}x{input_h}")

        # ---- RealSense ----
        print(f"[INFO] Camera calibrated, K:\n{K}")

        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            raise RuntimeError("No RealSense device found")

        print("[INFO] Resetting camera...")
        devices[0].hardware_reset()
        time.sleep(3)

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

        pipeline.start(config)
        print("[INFO] Camera started")

        while True:
            frames = pipeline.wait_for_frames()#cv2.imread("test_car_x60cm.png")
            color_frame = frames.get_color_frame()#cv2.imread("test_car_x60cm.png")

            if not color_frame:
               continue

            frame = np.asanyarray(color_frame.get_data())
            #frame = np.asanyarray(color_frame)

            display_frame = detect_lanes(frame) if ENABLE_LANE_DETECTION else frame.copy()

            # ---- PREPROCESS ----
            img = cv2.resize(frame, (input_w, input_h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.astype(np.float32) / 255.0

            # CHW + batch
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)

            img = np.ascontiguousarray(img)

            # ---- INFERENCE ----
            output = model.infer(img)

            print("Output sample:", output[:10])

            # ---- DECODE ----
            boxes, scores = decode(output, input_w, input_h)

            if len(boxes) > 0:
                keep = nms(boxes, scores, NMS_THRESH)

                if len(keep) > 0:
                    idx = keep[int(np.argmax(scores[keep]))]  # best score among kept boxes
                    box = boxes[idx]
                    score = scores[idx]

                    scale_x = frame.shape[1] / input_w
                    scale_y = frame.shape[0] / input_h

                    x1 = int(box[0] * scale_x)
                    y1 = int(box[1] * scale_y)
                    x2 = int(box[2] * scale_x)
                    y2 = int(box[3] * scale_y)

                    x1 = max(0, min(x1, frame.shape[1]))
                    y1 = max(0, min(y1, frame.shape[0]))
                    x2 = max(0, min(x2, frame.shape[1]))
                    y2 = max(0, min(y2, frame.shape[0]))

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{score:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # bottom center of box in original frame coords
                    u_bottom = (x1 + x2) // 2
                    v_bottom = y2

                    x_car, y_car = pixel_to_car(K, u_bottom, v_bottom, CAMERA_HEIGHT_H)
                    distance = np.sqrt(x_car**2 + y_car**2)

                    cv2.putText(display_frame, f"{score:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(display_frame, f"{distance:.2f}m", (x1, y2 + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    print(f"Distance: x={x_car:.3f}m  y={y_car:.3f}m  total={distance:.3f}m")

            # ---- DISPLAY ----
            cv2.imshow("Frame", display_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        pipeline.stop()

    finally:
        cuda_ctx.pop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
