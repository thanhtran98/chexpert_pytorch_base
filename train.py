from os.path import split
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from metrics import OA_loss
from sklearn import metrics
import time, os, cv2, shutil
from data.utils import transform
from tensorboardX import SummaryWriter
from model.models import ResNeSt_parallel, Efficient_parallel, Efficient, ResNeSt
from model.classifier import Classifier
from resnest.torch import resnest50, resnest101, resnest200, resnest269
from efficientnet_pytorch import EfficientNet
import tqdm

def build_parallel_model(model_name, id, cfg=None, pretrained=False, split_output=False, modify_gp=False):
    if model_name == 'resnest':
        childs_cut = 9
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
        if split_output:
            model = ResNeSt(pre_model, cfg, modify_gp)
        else:
            model = ResNeSt_parallel(pre_model, 5)
    else:
        childs_cut = 6
        pre_name = 'efficientnet-'+id
        if pretrained:
            pre_model = EfficientNet.from_pretrained(pre_name)
        else:
            pre_model = EfficientNet.from_name(pre_name)
        for param in pre_model.parameters():
            param.requires_grad = True
        if split_output:
            model = Efficient(pre_model, cfg, modify_gp)
        else:
            model = Efficient_parallel(pre_model, 5)
    return model, childs_cut

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
    id_obs = [2,5,6,8,10]
    id_leaf = [2,4,5,6,7,8]
    id_parent = [aa for aa in range(14) if aa not in id_leaf]
    M = np.array([[0,1,-2,0,0,0,0,0,0,0,0,0,0,0],
                  [0,0,0,1,-2,0,0,0,0,0,0,0,0,0],
                  [0,0,0,1,0,-2,0,0,0,0,0,0,0,0],
                  [0,0,0,1,0,0,-2,0,0,0,0,0,0,0],
                  [0,0,0,1,0,0,0,-2,0,0,0,0,0,0],
                  [0,0,0,1,0,0,0,0,-2,0,0,0,0,0]])
    
    def __init__(self, cfg, optimizer, origin=True, split_output=False, modify_gp=False, loss_func=None,
                 model_name='resnest', id='50', lr=3e-4, metrics=None, pretrained=True):
        self.model_name = model_name
        self.id = id
        self.cfg = cfg
        self.split_output = split_output
        self.modify_gp = modify_gp
        if origin:
            self.split_output = True
            self.modify_gp = True
            self.model = Classifier(self.cfg)
        else:
            self.model, self.childs_cut = build_parallel_model(self.model_name, self.id, self.cfg, pretrained, self.split_output, self.modify_gp)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.split_output:
            self.loss_func = OA_loss(self.device, self.cfg)
        else:
            self.loss_func = loss_func
        if metrics is not None:
            self.metrics = metrics
            self.metrics['loss'] = self.loss_func
        else:
            self.metrics = {'loss': self.loss_func}
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(),self.lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def freeze_head(self):
        ct = 0
        for child in self.model.children():
            ct += 1
            if ct < self.childs_cut:
                for param in child.parameters():
                    param.requires_grad = False
    
    def load_ckp(self, ckp_path):
        ckp = torch.load(ckp_path, map_location=self.device)
        self.model.load_state_dict(ckp['state_dict'])

    def load_resume_ckp(self, ckp_path):
        ckp = torch.load(ckp_path, map_location=self.device)
        self.model.load_state_dict(ckp['state_dict'])
        return ckp['epoch'], ckp['iter']
    
    def save_ckp(self, ckp_path, epoch, i):
        if os.path.exists(os.path.dirname(ckp_path)):
            torch.save(
                        {'epoch': epoch+1,
                        'iter': i+1,
                        'state_dict':self. model.state_dict()},
                        ckp_path
                        )
        else:
            print("Save path not exist!!!")
    
    def predict(self, image):

        torch.set_grad_enabled(False)
        self.model.eval()
        with torch.no_grad() as tng:
            preds = self.model(image)
            if self.split_output:
                preds = nn.Sigmoid()(torch.cat([aa for aa in preds], axis=-1))
                preds = preds.cpu().numpy()
                # preds = np.concatenate([aa.cpu().numpy() for aa in preds], axis=-1)
            else:
                preds = nn.Sigmoid()(preds)
                preds = preds.cpu().numpy()

        return preds

    def predict_from_file(self, image_file):

        image_gray = cv2.imread(image_file, 0)  
        image = transform(image_gray, self.cfg)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)

        return self.predict(image)

    def train(self, loader_dict, epochs=120, iter_log=100, use_lr_sch=False, resume=False, ckp_dir='./experiment/checkpoint',
              writer=None, eval_metric='loss', mix_precision=False, conditional_training=False):
        if 'train' not in list(loader_dict.keys()):
            raise Exception("missing \'train\' keys in loader_dict!!!")
        if use_lr_sch:
            lr_sch = torch.optim.lr_scheduler.StepLR(self.optimizer, int(epochs*2/3), self.lr/3)
        else:
            lr_sch = None
        if conditional_training:
            tranform2leaf_matrix = np.zeros((len(self.id_leaf), len(self.cfg.num_classes)))
            for i, leaf_id in enumerate(self.id_leaf):
                tranform2leaf_matrix[i][leaf_id] = 1.0
            tranform2leaf_matrix = torch.from_numpy(tranform2leaf_matrix).to(self.device)
        best_metric = 0.0
        if os.path.exists(ckp_dir) != True:
            os.mkdir(ckp_dir)
        if resume:
            epoch_resume, iter_resume = self.load_resume_ckp(os.path.join(ckp_dir,'latest.ckpt'))
        else:
            epoch_resume = 1
            iter_resume = 0
        modes = list(loader_dict.keys())
        history = dict.fromkeys(modes, {})
        if mix_precision:
            print('Train with mix precision!')
            scaler = torch.cuda.amp.GradScaler()
        for mode in modes:
            history[mode] = dict.fromkeys(self.metrics.keys(), [])
        for epoch in range(epoch_resume-1, epochs):
            start = time.time()
            for mode in modes:
                running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)
                if mode == 'train':
                    running_metrics.pop('auc', None)
                ova_len = loader_dict[mode].dataset._num_image
                n_iter = len(loader_dict[mode])
                # n_log = n_iter//iter_log + 1
                if mode == 'train':
                    torch.set_grad_enabled(True)
                    self.model.train()
                    batch_weights = (1/iter_log)*np.ones(n_iter)
                    if n_iter%iter_log:
                        batch_weights[-(n_iter%iter_log):] = 1/(n_iter%iter_log)
                else:
                    torch.set_grad_enabled(False)
                    self.model.eval()
                    batch_weights = (1/n_iter)*np.ones(n_iter)
                for i, data in tqdm.tqdm(enumerate(loader_dict[mode])):
                    imgs, labels = data[0].to(self.device), data[1].to(self.device)
                    if mix_precision:
                        with torch.cuda.amp.autocast():
                            preds = self.model(imgs)
                            if conditional_training:
                                preds = tranform2leaf_matrix.dot(preds)
                                labels = tranform2leaf_matrix.dot(labels)
                            loss = self.metrics['loss'](preds, labels)
                    else:
                        preds = self.model(imgs)
                        if conditional_training:
                            preds = tranform2leaf_matrix.dot(preds)
                            labels = tranform2leaf_matrix.dot(labels)
                        loss = self.metrics['loss'](preds, labels)
                    if self.split_output:
                        preds = torch.cat([aa for aa in preds], dim=-1)
                    preds = nn.Sigmoid()(preds)
                    running_loss = loss.item()*batch_weights[i]
                    for key in list(running_metrics.keys()):
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
                            metric_eval = running_metrics_test[eval_metric][self.id_obs]
                            s = s[:-1] + " - mean_"+eval_metric+" {:.3f}".format(metric_eval.mean())
                            self.save_ckp(os.path.join(ckp_dir,'latest.ckpt'), epoch, i)
                            if writer is not None:
                                for key in list(running_metrics.keys()):
                                    writer.add_scalars(key, {mode: running_metrics[key].mean()}, (epoch*n_iter)+(i+1))
                            running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)
                            running_metrics.pop('auc', None)
                            end = time.time()
                            s = s[:-1] + " ({:.1f}s)".format(end-start)
                            print(s)
                            if metric_eval.mean() > best_metric:
                                best_metric = metric_eval.mean()
                                shutil.copyfile(os.path.join(ckp_dir,'latest.ckpt'), os.path.join(ckp_dir,'epoch'+str(epoch+1)+'_iter'+str(i+1)+'.ckpt'))
                                print('new checkpoint saved!')
                            
                            start = time.time()
                if mode == 'train':
                    s="Epoch [{}/{}] Iter [{}/{}]:\n".format(epoch+1, epochs, n_iter, n_iter)
                s = get_str(running_metrics, mode, s)
            end = time.time()
            s = s + "({:.1f}s)".format(end-start)
            print(s)
            if lr_sch is not None:
                lr_sch.step()
                print('current lr: {:.4f}'.format(lr_sch.get_lr()[0]))
        return history

    def conditional_train(self):
        pass

    def test(self, loader, mix_precision=False, ensemble=False):
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
            if self.split_output:
                preds = torch.cat([aa for aa in preds], dim=-1)
            preds = nn.Sigmoid()(preds)
            if ensemble:
                preds = torch.mm(preds,self.ensemble_weights)
                labels = labels[:,self.id_obs]
                loss = nn.MSELoss()(preds, labels)
            iter_len = imgs.size()[0]
            if i == 0:
                preds_stack = preds
                labels_stack = labels
                running_loss = loss.item()*iter_len/ova_len
            else:
                preds_stack = torch.cat((preds_stack, preds), 0)
                labels_stack = torch.cat((labels_stack, labels), 0)
                running_loss += loss.item()*iter_len/ova_len
        for key in list(self.metrics.keys()):
            if key == 'loss':
                running_metrics[key] = running_loss
            else:
                running_metrics[key] = self.metrics[key](preds_stack, labels_stack).cpu().numpy()
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

    def ensemble(self, loader, test_loader, type='basic', init_lr=0.01, log_step=100):
        n_class = len(self.cfg.num_classes)
        w = np.random.rand(n_class, len(self.id_obs))
        iden_matrix = np.diag(np.ones(n_class))
        lr = init_lr
        step = log_step
        start = time.time()
        for i, data in enumerate(loader):
            imgs, labels = data[0].to(self.device), data[1]
            preds = self.predict(imgs)
            labels = labels.numpy()
            labels = labels[:,self.id_obs]
            if type=='basic':
                grad = (preds.T.dot(preds) + 0.1*iden_matrix).dot(w) - preds.T.dot(labels)
            elif type=='b_constraint':
                grad = (preds.T.dot(preds) + 0.1*iden_matrix + 2*self.M.T.dot(self.M)).dot(w) + - preds.T.dot(labels)
            else:
                raise Exception("Not support this type!!!")
            w -= lr*grad
            if (i+1)%step == 0:
                print(w)
                # lr = decayed_learning_rate(i+1, ilr, decay_steps=step)
                # print(lr)
                end = time.time()
                print('iter {:d} time takes: {:.3f}s'.format(i+1, end-start))
                start = time.time()
                if (i+1)//step==20:
                    break
        self.ensemble_weights = torch.from_numpy(w).float().to(self.device)
        print('Done Essemble!!!')
        metrics = self.test(test_loader, ensemble=True)
        return metrics