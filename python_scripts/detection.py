import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2


class TRTDetector:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Get input/output names
        self.input_name = None
        self.output_name = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        print("[TRT] Input:", self.input_name)
        print("[TRT] Output:", self.output_name)

        self.d_input = None
        self.d_output = None

    def preprocess(self, img):
        print("[INFO] Original shape:", img.shape)

        # MUST match engine input (480, 270)
        img = cv2.resize(img, (320, 180))

        print("[INFO] Resized shape:", img.shape)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)

        return np.ascontiguousarray(img)

    def infer(self, input_data):
        if self.d_input is None:
            self.context.set_input_shape(self.input_name, input_data.shape)

            input_size = input_data.size * 4
            self.d_input = cuda.mem_alloc(input_size)

            out_shape = self.context.get_tensor_shape(self.output_name)
            out_size = int(np.prod(out_shape)) * 4
            self.d_output = cuda.mem_alloc(out_size)

            print("[INFO] Allocated buffers")
            print("[INFO] Output shape:", out_shape)

        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)

        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))

        self.context.execute_async_v3(self.stream.handle)

        output_shape = self.context.get_tensor_shape(self.output_name)
        output = np.empty(output_shape, dtype=np.float32)

        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)

        self.stream.synchronize()

        print("[INFO] Inference done. Output shape:", output.shape)

        return output

    def detect(self, img):
        input_data = self.preprocess(img)
        output = self.infer(input_data)

        return output
