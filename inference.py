from PIL import Image
import torch
import requests

# fetch new version of MiDaS
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('NielsRogge/ZoeDepth:understanding_zoedepth', "ZoeD_N", pretrained=True, force_reload=True).to(DEVICE).eval()

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

depth = model.infer_pil(image)

raw_depth = Image.fromarray((depth*256).astype('uint16'))
raw_depth.save(".")