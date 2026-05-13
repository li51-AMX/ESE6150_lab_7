import argparse

import tensorrt as trt
import sys

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, fp16=True):
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print("Parsing ONNX model...")
    with open(onnx_file_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    print("Creating builder config...")
    config = builder.create_builder_config()

    # Set workspace memory (TensorRT 8+)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 1 << 28  # 256MB
    )

    # Enable FP16 if supported
    if fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16...")
        config.set_flag(trt.BuilderFlag.FP16)
    

    print("Building TensorRT engine...")

    # NEW API (replaces build_engine)
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None

    print("Creating runtime...")
    runtime = trt.Runtime(TRT_LOGGER)

    engine = runtime.deserialize_cuda_engine(serialized_engine)

    if engine is None:
        print("ERROR: Failed to deserialize engine")
        return None

    # Save engine
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Engine saved to {engine_file_path}")
    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX model")
    parser.add_argument("--onnx", default="model/model.onnx", help="Path to the ONNX model")
    parser.add_argument("--engine", required=True, help="Output path for the TensorRT engine")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 if supported")
    args = parser.parse_args()

    build_engine(args.onnx, args.engine, fp16=args.fp16)