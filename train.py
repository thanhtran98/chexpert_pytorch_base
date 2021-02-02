from data.dataset import ImageDataset, ImageDataset_full
from easydict import EasyDict as edict
from torch.utils.data import DataLoader
import json, os
from metrics import F1, ACC, AUC
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from model.chexpert import CheXpert_model

cfg_path = './config/example.json'

with open(cfg_path) as f:
    cfg = edict(json.load(f))

# loss_func = BCELoss()
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
loader_dict = {'train': train_loader, 'val': val_loader}

chexpert_model = CheXpert_model(cfg, loss_func, metrics_dict)

# chexpert_model.load_ckp(cfg.ckp_path)
# chexpert_model.freeze_backbone()

writer = SummaryWriter(os.path.join('experiment', cfg.log_dir))
ckp_dir = os.path.join('experiment', cfg.log_dir, 'checkpoint')

chexpert_model.train(train_loader, val_loader, epochs=cfg.epochs, iter_log=cfg.iter_log, writer=writer, eval_metric='auc', ckp_dir=ckp_dir)