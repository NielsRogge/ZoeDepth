from PIL import Image
import torch
import requests

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to(DEVICE).eval()

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

depth = model.infer_pil(image)

raw_depth = Image.fromarray((depth*256).astype('uint16'))
raw_depth.save(".")