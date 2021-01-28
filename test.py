import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from data.dataset import ImageDataset, ImageDataset_full
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import json, os, shutil, torch
import torch.nn as nn
from train import ChexPert_model
from metrics import F1, ACC, AUC
from torch.optim import Adam
from torch.nn import BCELoss, BCEWithLogitsLoss
from torchvision.models import densenet121
from torch.utils.tensorboard import SummaryWriter

cfg_path = './config/example_PCAM.json'
train_csv = './CheXpert-v1.0-small/train.csv'
val_csv = './CheXpert-v1.0-small/valid.csv'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

optimizer = Adam
loss_func = BCEWithLogitsLoss()

split_output=True
modify_gp=False
mix_precision=True
origin=False
pretrained=False
full_classes=True
conditional_training=False
lr=1e-4

batch_size = 64

if full_classes:
    data_class = ImageDataset_full
else:
    data_class = ImageDataset

train_loader = DataLoader(data_class(train_csv, cfg, mode='train'),
                          num_workers=4,drop_last=True,shuffle=True,
                          batch_size=batch_size)
val_loader = DataLoader(data_class(val_csv, cfg, mode='dev'),
                        num_workers=4,drop_last=False,shuffle=False,
                        batch_size=batch_size)

metrics_dict = {'acc': ACC(), 'auc':AUC()}
# loader_dict = {'train': train_loader, 'val': val_loader}

# model_names=['resnest', 'efficient', 'dense', 'resnest']
# ids = ['50', 'b4', '121', '101']
# ckp_paths = [
#     'experiment/train_log/ResNeSt50/21h42_130121 (end)/best1.ckpt',
#     'experiment/train_log/EfficientNet/23h_130121 (end)/best1.ckpt',
#     'experiment/train_log/DenseNet121/21h40_140121 (end)/best1.ckpt',
#     'experiment/train_log/ResNeSt101/1h_160121 (end)/best1.ckpt'
#     ]

# ckp_paths = ['experiment/train_log/ResNeSt_parallel/epoch2_iter5400.ckpt',
#              'experiment/train_log/EfficientNet_parallel/20h52_140121 (end)/epoch3_iter3200.ckpt',
#              'experiment/train_log/DenseNet121-parallel/epoch1_iter7000.ckpt',
#              'experiment/train_log/ResNeSt101-parallel/epoch2_iter7600.ckpt'
#             ]

# ckp_paths = [
#     'experiment/train_log/ResNeSt50/21h_210121_14class/epoch2_iter200.ckpt',
#     'experiment/train_log/EfficientNet/23h_210121_14classes/epoch1_iter1600.ckpt',
#     'experiment/train_log/DenseNet121/23h_230121_14classes/epoch3_iter800.ckpt',
#     'experiment/train_log/ResNeSt101/23h_210121_14class/epoch3_iter400.ckpt'
#     ]

model_names = ['dense']
ids = ['121']
# ckp_paths = ['experiment/DenseNet121_conditional_finetune_2601/checkpoint/epoch5_iter200.ckpt']
ckp_paths = ['experiment/DenseNet121_conditional_finetune_2601_14/checkpoint/epoch10_iter400.ckpt']

id_leaf = [2,4,5,6,7,8]
id_obs = [2,5,6,8,10]

for i, model_name in enumerate(model_names):
    chexpert_model = ChexPert_model(cfg, optimizer, origin, split_output, modify_gp, loss_func,
                                    model_name, ids[i], lr, metrics_dict, pretrained, full_classes)
    chexpert_model.load_ckp(ckp_paths[i])
    metrics = chexpert_model.test(val_loader, mix_precision, conditional_training=conditional_training)
    if full_classes and not conditional_training:
        print(model_name+'-'+ids[i]+':\n', metrics['auc'][id_obs], metrics['auc'][id_obs].mean(), '\n', metrics['acc'][id_obs], metrics['acc'][id_obs].mean())
    else:
        print(model_name+'-'+ids[i]+':\n', metrics['auc'], metrics['auc'].mean(), '\n', metrics['acc'], metrics['acc'].mean())