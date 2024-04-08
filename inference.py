from PIL import Image
import torch
import requests

from huggingface_hub import hf_hub_download

from zoedepth.utils.misc import colorize

# fetch new version of MiDaS
torch.hub.help("NielsRogge/MiDaS:fix_beit_backbone", "DPT_BEiT_L_384", force_reload=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('NielsRogge/ZoeDepth:understanding_zoedepth', "ZoeD_NK", pretrained=True, force_reload=True).to(DEVICE).eval()

# cats image (used for comparing logits)
# resize to square (our HF beit implementation does not support arbitrary resolutions yet)
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# image = image.resize((384, 384))

filepath = hf_hub_download(repo_id="shariqfarooq/ZoeDepth", filename="examples/person_1.jpeg", repo_type="space")
image = Image.open(filepath).convert("RGB")

# we also specify pad=input=False when comparing logits
depth = model.infer_pil(image, pad_input=False, with_flip_aug=False)

print("Shape of predicted depth:", depth.shape)

colored_depth = colorize(depth, cmap='gray_r')
raw_depth = Image.fromarray((depth*256).astype('uint16'))
raw_depth.save("predicted_depth.png")

img = Image.fromarray(colored_depth, mode='RGBA')
img.save("predicted_depth_colorized.png")