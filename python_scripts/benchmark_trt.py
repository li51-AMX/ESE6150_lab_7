#!/usr/bin/env python3

import argparse
import os
import time

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTModel:
    def __init__(self, engine_path):
        logger = TRT_LOGGER

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

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

        if not self.inputs or not self.outputs:
            raise RuntimeError("Engine must have at least one input and one output tensor")

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


def load_and_preprocess(image_path, input_w, input_h):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    resized = cv2.resize(image, (input_w, input_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    batched = np.expand_dims(chw, axis=0)
    return np.ascontiguousarray(batched)


def benchmark(engine_path, image_path, warmup=20, runs=100):
    cuda.init()
    cuda_ctx = cuda.Device(0).make_context()

    try:
        model = TRTModel(engine_path)

        name = model.engine.get_tensor_name(0)
        _, _, input_h, input_w = model.engine.get_tensor_shape(name)

        print(f"[INFO] Engine: {engine_path}")
        print(f"[INFO] Input size: {input_w}x{input_h}")
        print(f"[INFO] Image: {image_path}")

        img = load_and_preprocess(image_path, input_w, input_h)

        for _ in range(warmup):
            _ = model.infer(img)

        timings = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model.infer(img)
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)

        avg_ms = sum(timings) / len(timings)
        print(f"[RESULT] Average inference time over {runs} runs: {avg_ms:.3f} ms")
        return avg_ms

    finally:
        cuda_ctx.pop()


def main():
    parser = argparse.ArgumentParser(description="Benchmark TensorRT inference time")
    parser.add_argument("--engine", required=True, help="Path to .trt engine file")
    parser.add_argument(
        "--image",
        default="test_car_x60cm.png",
        help="Image to preprocess and reuse for timing",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Timed iterations")
    args = parser.parse_args()

    engine_path = args.engine
    image_path = args.image

    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine not found: {engine_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    benchmark(engine_path, image_path, warmup=args.warmup, runs=args.runs)


if __name__ == "__main__":
    main()
