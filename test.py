import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from data.dataset import ImageDataset, ImageDataset_full
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import json, os
from model.chexpert import CheXpert_model
from metrics import F1, ACC, AUC
from torch.optim import Adam
from torch.nn import BCELoss, BCEWithLogitsLoss

cfg_path = './config/example.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

optimizer = Adam
loss_func = BCEWithLogitsLoss()

if cfg.full_classes:
    data_class = ImageDataset_full
else:
    data_class = ImageDataset

train_loader = DataLoader(data_class(cfg.train_csv, cfg, mode='train'),
                          num_workers=4,drop_last=True,shuffle=True,
                          batch_size=cfg.batch_size)
val_loader = DataLoader(data_class(cfg.dev_csv, cfg, mode='dev'),
                        num_workers=4,drop_last=False,shuffle=False,
                        batch_size=cfg.batch_size)

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

id_leaf = [2,4,5,6,7,8]
id_obs = [2,5,6,8,10]

chexpert_model = CheXpert_model(cfg, optimizer, loss_func, metrics_dict)
chexpert_model.load_ckp(cfg.ckp_path)
metrics = chexpert_model.test(val_loader)
if cfg.full_classes and not cfg.conditional_training:
    print(cfg.model_name+'-'+cfg.id+':\n', metrics['auc'][id_obs], metrics['auc'][id_obs].mean(), '\n', metrics['acc'][id_obs], metrics['acc'][id_obs].mean())
else:
    print(cfg.model_name+'-'+cfg.id+':\n', metrics['auc'], metrics['auc'].mean(), '\n', metrics['acc'], metrics['acc'].mean())