import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
import os

ONNX_FILE_PATH = "model/model_v3.onnx"
ENGINE_PATH = "model/model_v4.trt"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

INPUT_SHAPE = (1, 3, 180, 320)

# =========================
# PREPROCESS
# =========================
def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    img = cv2.resize(img, (INPUT_SHAPE[3], INPUT_SHAPE[2]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return np.ascontiguousarray(img, dtype=np.float32)


# =========================
# POSTPROCESS
# =========================
def postprocess(output, conf_thresh=0.5):
    output = output[0]

    conf = output[0]
    dx   = output[1]
    dy   = output[2]
    w    = output[3]
    h    = output[4]

    H, W = conf.shape

    boxes = []
    scores = []

    for i in range(H):
        for j in range(W):

            if conf[i, j] < conf_thresh:
                continue

            cell_w = INPUT_SHAPE[3] / W
            cell_h = INPUT_SHAPE[2] / H

            cx = j * cell_w + cell_w / 2 + dx[i, j]
            cy = i * cell_h + cell_h / 2 + dy[i, j]

            bw = w[i, j] * INPUT_SHAPE[3]
            bh = h[i, j] * INPUT_SHAPE[2]

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            boxes.append([x1, y1, x2, y2])
            scores.append(conf[i, j])

    return np.array(boxes), np.array(scores)


# =========================
# BUILD ENGINE
# =========================
def build_engine(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print("Building engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Engine saved to {engine_path}")

    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(serialized_engine)


# =========================
# LOAD ENGINE
# =========================
def load_engine(engine_path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


# =========================
# MAIN
# =========================
def main():
    # Load or build engine
    if os.path.exists(ENGINE_PATH):
        print("Loading engine...")
        engine = load_engine(ENGINE_PATH)
    else:
        engine = build_engine(ONNX_FILE_PATH, ENGINE_PATH)

    context = engine.create_execution_context()

    # Get tensor names
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

    input_name = None
    output_names = []

    for name in tensor_names:
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_names.append(name)

    # Set input shape
    context.set_input_shape(input_name, INPUT_SHAPE)

    # Get output shapes
    output_shapes = [context.get_tensor_shape(name) for name in output_names]

    print("Outputs:")
    for name, shape in zip(output_names, output_shapes):
        print(name, shape)

    # Allocate memory
    d_input = cuda.mem_alloc(trt.volume(INPUT_SHAPE) * np.float32().itemsize)

    d_outputs = []
    h_outputs = []

    for shape in output_shapes:
        size = trt.volume(shape)
        d_outputs.append(cuda.mem_alloc(size * np.float32().itemsize))
        h_outputs.append(cuda.pagelocked_empty(size, dtype=np.float32))

    stream = cuda.Stream()

    # Preprocess
    host_input = preprocess("test_car_x60cm.png")

    cuda.memcpy_htod_async(d_input, host_input, stream)

    # Bind tensors
    context.set_tensor_address(input_name, int(d_input))

    for name, d_out in zip(output_names, d_outputs):
        context.set_tensor_address(name, int(d_out))

    # Run inference
    context.execute_async_v3(stream_handle=stream.handle)

    # Copy outputs
    for h_out, d_out in zip(h_outputs, d_outputs):
        cuda.memcpy_dtoh_async(h_out, d_out, stream)

    stream.synchronize()

    # Use FIRST output (your YOLO tensor)
    output = h_outputs[0].reshape(output_shapes[0])

    boxes, scores = postprocess(output)

    print("Boxes:", boxes)
    print("Scores:", scores)


if __name__ == "__main__":
    main()
