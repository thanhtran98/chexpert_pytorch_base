import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch, json
from efficientnet_pytorch import EfficientNet
from torchvision.models.densenet import densenet121
from resnest.torch import resnest50, resnest101, resnest200, resnest269, resnest50_fast_4s2x40d
from torchsummary import summary
import torch.nn as nn
from metrics import AUC
from model.models import Efficient, ResNeSt, ResNeSt_parallel, Efficient_parallel
from easydict import EasyDict as edict

cfg_path = './config/example_PCAM.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pre_model = densenet121(pretrained=False)
# pre_model = resnest50(pretrained=False)
pre_model = EfficientNet.from_pretrained('efficientnet-b4')
for param in pre_model.parameters():
  param.requires_grad = True

model = Efficient(pre_model, cfg, modify_gp=True)
# model = ResNeSt(pre_model, cfg)
# model = ResNeSt_parallel(pre_model, 5)
# model = Efficient_parallel(pre_model, 5)
# model = ResNeSt(pre_model, cfg, modify_gp=True)

dummy_input = torch.randn(3, 3, 512, 512)
# output = nn.Sigmoid()(model(dummy_input))
output = model(dummy_input)
print(output[0].shape)
# dummy_output = (nn.Sigmoid()(torch.randn(3, 5)) > 0.5)*1.0
# print(dummy_output)
# auc = AUC()
# print(auc(output, dummy_output))
# summary(model)
# print(pre_model)