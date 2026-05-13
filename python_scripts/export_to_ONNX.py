import torch
from model.f110_yolo_hw_v3 import F110_YOLO

model = F110_YOLO()
model.load_state_dict(torch.load("model/model_v3.pt")) 
model.eval()

dummy_input = torch.randn(1, 3, 320, 180)

torch.onnx.export(
    model,
    dummy_input,
    "model/model_v3.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("ONNX model saved as model_v3.onnx")