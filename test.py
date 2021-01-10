import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch, json
from models import ResNeSt
from efficientnet_pytorch import EfficientNet
from resnest.torch import resnest50, resnest101, resnest200, resnest269, resnest50_fast_4s2x40d
from torchsummary import summary
from models import Efficient, ResNeSt
from easydict import EasyDict as edict

cfg_path = './config/example_PCAM.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pre_model = resnest50(pretrained=False)
pre_model = EfficientNet.from_pretrained('efficientnet-b4')
for param in pre_model.parameters():
  param.requires_grad = True

model = Efficient(pre_model, cfg)
# model = ResNeSt(pre_model, cfg)

dummy_input = torch.randn(3, 3, 512, 512)
output = model(dummy_input)
print(output[0].shape)

summary(model)