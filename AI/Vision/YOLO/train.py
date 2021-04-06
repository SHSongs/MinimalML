import torch
from model import Yolov1


DEVICE = "cuda" if torch.cuda.is_available else "cpu"

model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
print(model)

output = model(torch.randn(1, 3, 448, 448).cuda())
print(output.shape)

predictions = output.reshape(-1, 7, 7, 20 + 2 * 5)
print(predictions.shape)