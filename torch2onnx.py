import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch, json
from efficientnet_pytorch import EfficientNet
from torchvision.models.densenet import densenet121
from resnest.torch import resnest50, resnest101, resnest200, resnest269, resnest50_fast_4s2x40d
from torchsummary import summary
from models import Efficient, ResNeSt, ResNeSt_parallel, Efficient_parallel
from models import Efficient_hm
from easydict import EasyDict as edict

cfg_path = './config/example_PCAM.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

# pre_model = densenet121(pretrained=False)
# pre_model = resnest50(pretrained=False)
pre_model = EfficientNet.from_pretrained('efficientnet-b4')
for param in pre_model.parameters():
  param.requires_grad = True

ckpt_path = 'best1_efficient.ckpt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(ckpt_path, map_location=device)
# model = ResNeSt(pre_model, cfg)
pre_model.set_swish(False)
model = Efficient_hm(pre_model, cfg)
# model = Efficient(pre_model, cfg)
model.load_state_dict(ckpt['state_dict'])

dummy_input = torch.randn(1, 3, 512, 512)

dummy_input = dummy_input.float()
# print(model)
input_names = ["input0"]
torch.onnx.export(model, dummy_input, "efficient-b4_base_heatmap.onnx", opset_version=11,
export_params=True, verbose=False, input_names=input_names)