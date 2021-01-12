import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from sklearn import metrics
import time, os, cv2, shutil
from data.utils import transform
from tensorboardX import SummaryWriter
from models import ResNeSt_parallel, Efficient_parallel
from resnest.torch import resnest50, resnest101, resnest200, resnest269
from efficientnet_pytorch import EfficientNet

def build_parallel_model(model_name, id, pretrained=False):
    if model_name == 'resnest':
        if id == '50':
            pre_name = resnest50
        elif id == '101':
            pre_name = resnest101
        elif id == '200':
            pre_name = resnest200
        else:
            pre_name = resnest269
        pre_model = pre_name(pretrained=pretrained)
        # pre_model = EfficientNet.from_pretrained('efficientnet-b4')
        for param in pre_model.parameters():
            param.requires_grad = True
        model = ResNeSt_parallel(pre_model, 5)
    else:
        pre_name = 'efficientnet-'+id
        pre_model = EfficientNet.from_pretrained(pre_name)
        for param in pre_model.parameters():
            param.requires_grad = True
        model = Efficient_parallel(pre_model, 5)
    return model

def get_str(metrics, mode, s):
    for key in list(metrics.keys()):
        if key == 'loss':
            s += "{}_{} {:.3f} - ".format(mode, key, metrics[key])
        else:
            metric_str = ' '.join(map(lambda x: '{:.5f}'.format(x), metrics[key]))
            s += "{}_{} {} - ".format(mode, key, metric_str)
    s = s[:-2] + '\n'
    return s

class ChexPert_model():
    disease_classes = [
    'Cardiomegaly',
    'Edema',
    'Consolidation',
    'Atelectasis',
    'Pleural Effusion'
    ]
    def __init__(self, cfg, loss_func, optimizer, model_name='resnest', lr=3e-4, metrics=None, pretrained=True):
        self.model_name = model_name
        self.cfg = cfg
        self.model = build_parallel_model(self.model_name, pretrained=pretrained)
        self.loss_func = loss_func
        if metrics is not None:
            self.metrics = metrics
            self.metrics['loss'] = loss_func
        else:
            self.metrics = {'loss': loss_func}
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(),self.lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def load_ckp(self, ckp_path):
        checkpoint=torch.load(ckp_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
    
    def save_ckp(self, ckp_path):
        if os.path.exists(os.path.dirname(ckp_path)):
            torch.save(self.model.state_dict(), ckp_path)
        else:
            print("Save path not exist!!!")
    
    def predict(self, image):

        torch.set_grad_enabled(False)
        self.model.eval()
        with torch.no_grad() as tng:
            preds = self.model(image)
            preds = preds.cpu().numpy()

        return preds

    def predict_from_file(self, image_file):

        image_gray = cv2.imread(image_file, 0)  
        image = transform(image_gray, self.cfg)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)

        return self.predict(image)

    def train(self, loader_dict, epochs=120, iter_log=100, use_lr_sch=False, ckp_dir='./checkpoint', writer=None, eval_metric='loss', mix_precision=False):
        if 'train' not in list(loader_dict.keys()):
            raise Exception("missing \'train\' keys in loader_dict!!!")
        if use_lr_sch:
            lr_sch = torch.optim.lr_scheduler.StepLR(self.optimizer, int(epochs*2/3), self.lr/3)
        else:
            lr_sch = None
        
        best_metric = 0.0
        if os.path.exists(ckp_dir):
            shutil.rmtree(ckp_dir)
        os.mkdir(ckp_dir)
        modes = list(loader_dict.keys())
        history = dict.fromkeys(modes, {})
        if mix_precision:
            print('Train with mix precision!')
            scaler = torch.cuda.amp.GradScaler()
        for mode in modes:
            history[mode] = dict.fromkeys(self.metrics.keys(), [])
        for epoch in range(epochs):
            start = time.time()
            for mode in modes:
                running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)
                ova_len = loader_dict[mode].dataset._num_image
                n_iter = len(loader_dict[mode])
                # n_log = n_iter//iter_log + 1
                if mode == 'train':
                    torch.set_grad_enabled(True)
                    self.model.train()
                    batch_weights = (1/iter_log)*np.ones(n_iter)
                    batch_weights[-(n_iter%iter_log):] = 1/(n_iter%iter_log)
                else:
                    torch.set_grad_enabled(False)
                    self.model.eval()
                    batch_weights = (1/n_iter)*np.ones(n_iter)
                for i, data in enumerate(loader_dict[mode]):
                    imgs, labels = data[0].to(self.device), data[1].to(self.device)
                    if mix_precision:
                        with torch.cuda.amp.autocast():
                            # loss = model(data)
                            preds = self.model(imgs)
                            loss = self.metrics['loss'](preds, labels)
                    else:
                        preds = self.model(imgs)
                        loss = self.metrics['loss'](preds, labels)
                    running_loss = loss.item()*batch_weights[i]
                    for key in list(self.metrics.keys()):
                        if key == 'loss':
                            running_metrics[key] += running_loss
                        else:
                            running_metrics[key] += self.metrics[key](preds, labels).cpu().numpy()*batch_weights[i]
                    if mode == 'train':
                        self.optimizer.zero_grad()
                        if mix_precision:
                            scaler.scale(loss).backward()
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()
                        if (i+1)%iter_log == 0:
                            s="Epoch [{}/{}] Iter [{}/{}]:\n".format(epoch+1, epochs, i+1, n_iter)
                            s = get_str(running_metrics, mode, s)
                            running_metrics_test = self.test(loader_dict[modes[-1]], mix_precision)
                            s = get_str(running_metrics_test, modes[-1], s)
                            if running_metrics_test[eval_metric].mean() > best_metric:
                                best_metric = running_metrics_test[eval_metric].mean()
                                torch.save(self.model.state_dict(), os.path.join(ckp_dir,'epoch'+str(epoch+1)+'.pt'))
                                print('new checkpoint saved!')
                            if writer is not None:
                                for key in list(self.metrics.keys()):
                                    writer.add_scalars(key, {mode: running_metrics[key].mean()}, epoch)
                            running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)
                            end = time.time()
                            s = s[:-1] + " ({:.1f}s)".format(end-start)
                            print(s)
                            start = time.time()
                if mode == 'train':
                    s="Epoch [{}/{}] Iter [{}/{}]:\n".format(epoch+1, epochs, n_iter, n_iter)
                s = get_str(running_metrics, mode, s)
            end = time.time()
            s = s[:-1] + "({:.1f}s)".format(end-start)
            print(s)
            if lr_sch is not None:
                lr_sch.step()
                print('current lr: {:.4f}'.format(lr_sch.get_lr()[0]))
        return history

    def test(self, loader, mix_precision=False):
        torch.set_grad_enabled(False)
        self.model.eval()
        running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)
        ova_len = loader.dataset._num_image
        for i, data in enumerate(loader):
            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            if mix_precision:
                with torch.cuda.amp.autocast():
                    preds = self.model(imgs)
                    loss = self.metrics['loss'](preds, labels)
            else:
                preds = self.model(imgs)
                loss = self.metrics['loss'](preds, labels)
            iter_len = imgs.size()[0]
            running_loss = loss.item()*iter_len/ova_len
            for key in list(self.metrics.keys()):
                if key == 'loss':
                    running_metrics[key] += running_loss
                else:
                    running_metrics[key] += self.metrics[key](preds, labels).cpu().numpy()*iter_len/ova_len
        torch.set_grad_enabled(True)
        self.model.train()
        return running_metrics

    def evaluate(self, loader):
        torch.set_grad_enabled(False)
        self.model.eval()
        with torch.no_grad() as tng:
            ova_len = loader.dataset._num_image
            running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)
            for i, data in enumerate(loader):
                imgs, targets = data[0].to(self.device), data[1].to(self.device)
                preds = self.model(imgs)
                iter_len = imgs.size()[0]
                for key in list(self.metrics.keys()):
                    running_metrics[key] += self.metrics[key](preds, targets).item()*iter_len/ova_len
        s=""
        for key in list(self.metrics.keys()):
            s += "{}: {:.3f} - ".format(key, running_metrics[key])
        print(s[:-2])
        return running_metrics