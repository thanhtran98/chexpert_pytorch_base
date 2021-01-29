from torch import tensor
import torch.nn as nn
import numpy as np
import time
import cv2
import os
import shutil
import tqdm
import pickle
import torch
from data.utils import transform
from model.utils import get_model, get_str, tensor2numpy
from model.classifier import Classifier


class CheXpert_model():
    disease_classes = [
        'Cardiomegaly',
        'Edema',
        'Consolidation',
        'Atelectasis',
        'Pleural Effusion'
    ]
    id_obs = [2, 5, 6, 8, 10]
    id_leaf = [2, 4, 5, 6, 7, 8]
    id_parent = [0, 1, 3, 9, 10, 11, 12, 13]
    M = np.array([[0, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0]])

    def __init__(self, cfg, optimizer, loss_func, split_output=False, modify_gp=False,
                 model_name='resnest', id='50', lr=3e-4, metrics=None, pretrained=True, full_classes=False):
        """CheXpert class contains all functions used for training and testing our models

        Args:
            cfg (str): path to .json file contains model configuration.
            optimizer (torch.optim): optimizer of the model.
            loss_func (torch.nn.Module): loss function of the model.
            split_output (bool, optional): split the head after the model backbone into branches. Defaults to False.
            modify_gp (bool, optional): use another gobal pooling (PCAM/LSE/EXP/AVG). Defaults to False.
            model_name (str, optional): name of the model (resnest/efficientnet/densenet). Defaults to 'resnest'.
            id (str, optional): model version (eg: 50 - resnest50, b4 - efficientnet-b4). Defaults to '50'.
            lr ([type], optional): initial value of learning rate. Defaults to 3e-4.
            metrics ([type], optional): metrics use to evaluate model performance. Defaults to None.
            pretrained (bool, optional): use ImageNet pretrained weights. Defaults to True.
            full_classes (bool, optional): training with full 14 classes of observations. Defaults to False.
        """
        self.model_name = model_name
        self.id = id
        self.cfg = cfg
        self.full_classes = full_classes
        if self.full_classes:
            self.cfg.num_classes = 14*[1]
        self.split_output = split_output
        self.modify_gp = modify_gp
        self.model, self.childs_cut = get_model(
            self.model_name, self.id, self.cfg, pretrained, self.split_output, self.modify_gp)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_func = loss_func
        if metrics is not None:
            self.metrics = metrics
            self.metrics['loss'] = self.loss_func
        else:
            self.metrics = {'loss': self.loss_func}
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), self.lr)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def freeze_backbone(self):
        """Freeze model backbone
        """
        ct = 0
        for child in self.model.children():
            ct += 1
            if ct < self.childs_cut:
                for param in child.parameters():
                    param.requires_grad = False

    def load_ckp(self, ckp_path):
        """Load checkpoint

        Args:
            ckp_path (str): path to checkpoint

        Returns:
            int, int: current epoch, current iteration
        """
        ckp = torch.load(ckp_path, map_location=self.device)
        self.model.load_state_dict(ckp['state_dict'])

        return ckp['epoch'], ckp['iter']

    def save_ckp(self, ckp_path, epoch, i):
        """Save checkpoint

        Args:
            ckp_path (str): path to saved checkpoint
            epoch (int): current epoch
            i (int): current iteration
        """
        if os.path.exists(os.path.dirname(ckp_path)):
            torch.save(
                {'epoch': epoch+1,
                 'iter': i+1,
                 'state_dict': self. model.state_dict()},
                ckp_path
            )
        else:
            print("Save path not exist!!!")

    def predict(self, image, mix_precision=False, conditional_training=False):
        """Run prediction

        Args:
            image (torch.Tensor): images to predict. Shape (batch size, C, H, W)
            mix_precision (bool, optional): use mix percision. Defaults to False.
            conditional_training (bool, optional): use conditional training. Defaults to False.

        Returns:
            torch.Tensor: model prediction
        """
        torch.set_grad_enabled(False)
        self.model.eval()
        with torch.no_grad() as tng:
            if mix_precision:
                with torch.cuda.amp.autocast():
                    preds = self.model(image)
                    if self.split_output:
                        preds = torch.cat([aa for aa in preds], dim=-1)
                    if conditional_training:
                        preds = preds[:, self.id_leaf]
            else:
                preds = self.model(image)
                if self.split_output:
                    preds = torch.cat([aa for aa in preds], dim=-1)
                if conditional_training:
                    preds = preds[:, self.id_leaf]

        return preds

    def predict_from_file(self, image_file):
        """Run prediction from image path

        Args:
            image_file (str): image path

        Returns:
            numpy: model prediction in numpy array type
        """
        image_gray = cv2.imread(image_file, 0)
        image = transform(image_gray, self.cfg)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)

        return tensor2numpy(nn.Sigmoid()(self.predict(image)))

    def predict_loader(self, loader, mix_precision=False, ensemble=False, conditional_training=False):

        preds_stack = None
        labels_stack = None

        for i, data in enumerate(loader):
            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            preds = self.predict(imgs, mix_precision, conditional_training)
            if conditional_training:
                labels = labels[:, self.id_leaf]
            preds = nn.Sigmoid()(preds)
            if ensemble:
                preds = torch.mm(preds, self.ensemble_weights)
                labels = labels[:, self.id_obs]
            if i == 0:
                preds_stack = preds
                labels_stack = labels
            else:
                preds_stack = torch.cat((preds_stack, preds), 0)
                labels_stack = torch.cat((labels_stack, labels), 0)

        return preds_stack, labels_stack

    def train(self, train_loader, val_loader, epochs=120, iter_log=100, use_lr_sch=False, resume=False, ckp_dir='./experiment/checkpoint',
              writer=None, eval_metric='loss', mix_precision=False, conditional_training=False):
        if use_lr_sch:
            lr_sch = torch.optim.lr_scheduler.StepLR(
                self.optimizer, int(epochs*2/3), self.lr/3)
        else:
            lr_sch = None
        best_metric = 0.0
        if os.path.exists(ckp_dir) != True:
            os.mkdir(ckp_dir)
        if resume:
            epoch_resume, iter_resume = self.load_ckp(
                os.path.join(ckp_dir, 'latest.ckpt'))
        else:
            epoch_resume = 1
            iter_resume = 0
        scaler = None
        if mix_precision:
            print('Train with mix precision!')
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(epoch_resume-1, epochs):
            start = time.time()
            running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)
            running_metrics.pop(eval_metric, None)
            ova_len = train_loader.dataset._num_image
            n_iter = len(train_loader)
            # n_log = n_iter//iter_log + 1
            torch.set_grad_enabled(True)
            self.model.train()
            batch_weights = (1/iter_log)*np.ones(n_iter)
            if n_iter % iter_log:
                batch_weights[-(n_iter % iter_log):] = 1 / (n_iter % iter_log)
            for i, data in tqdm.tqdm(enumerate(train_loader)):
                imgs, labels = data[0].to(self.device), data[1].to(self.device)
                if mix_precision:
                    with torch.cuda.amp.autocast():
                        preds = self.model(imgs)
                        if self.split_output:
                            preds = torch.cat([aa for aa in preds], dim=-1)
                        if conditional_training:
                            preds = preds[:, self.id_leaf]
                            labels = labels[:, self.id_leaf]
                        loss = self.metrics['loss'](preds, labels)
                else:
                    preds = self.model(imgs)
                    if self.split_output:
                        preds = torch.cat([aa for aa in preds], dim=-1)
                    if conditional_training:
                        preds = preds[:, self.id_leaf]
                        labels = labels[:, self.id_leaf]
                    loss = self.metrics['loss'](preds, labels)
                preds = nn.Sigmoid()(preds)
                running_loss = loss.item()*batch_weights[i]
                for key in list(running_metrics.keys()):
                    if key == 'loss':
                        running_metrics[key] += running_loss
                    else:
                        running_metrics[key] += tensor2numpy(self.metrics[key](
                            preds, labels))*batch_weights[i]
                self.optimizer.zero_grad()
                if mix_precision:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                if (i+1) % iter_log == 0:
                    s = "Epoch [{}/{}] Iter [{}/{}]:\n".format(
                        epoch+1, epochs, i+1, n_iter)
                    s = get_str(running_metrics, 'train', s)
                    running_metrics_test = self.test(
                        val_loader, mix_precision, False, conditional_training)
                    s = get_str(running_metrics_test, 'val', s)
                    if conditional_training or (not self.full_classes):
                        metric_eval = running_metrics_test[eval_metric]
                    else:
                        metric_eval = running_metrics_test[eval_metric][self.id_obs]
                    s = s[:-1] + " - mean_"+eval_metric + \
                        " {:.3f}".format(metric_eval.mean())
                    self.save_ckp(os.path.join(
                        ckp_dir, 'latest.ckpt'), epoch, i)
                    if writer is not None:
                        for key in list(running_metrics.keys()):
                            writer.add_scalars(
                                key, {'train': running_metrics[key].mean()}, (epoch*n_iter)+(i+1))
                    running_metrics = dict.fromkeys(
                        self.metrics.keys(), 0.0)
                    running_metrics.pop(eval_metric, None)
                    end = time.time()
                    s = s[:-1] + " ({:.1f}s)".format(end-start)
                    print(s)
                    if metric_eval.mean() > best_metric:
                        best_metric = metric_eval.mean()
                        shutil.copyfile(os.path.join(ckp_dir, 'latest.ckpt'), os.path.join(
                            ckp_dir, 'epoch'+str(epoch+1)+'_iter'+str(i+1)+'.ckpt'))
                        print('new checkpoint saved!')
                    start = time.time()

            s = "Epoch [{}/{}] Iter [{}/{}]:\n".format(
                epoch+1, epochs, n_iter, n_iter)
            running_metrics = self.test(
                val_loader, mix_precision, conditional_training=conditional_training)
            s = get_str(running_metrics, 'val', s)
            if conditional_training or (not self.full_classes):
                metric_eval = running_metrics[eval_metric]
            else:
                metric_eval = running_metrics[eval_metric][self.id_obs]
            s = s[:-1] + " - mean_"+eval_metric + \
                " {:.3f}".format(metric_eval.mean())
            end = time.time()
            s = s + "({:.1f}s)".format(end-start)
            print(s)
            if lr_sch is not None:
                lr_sch.step()
                print('current lr: {:.4f}'.format(lr_sch.get_lr()[0]))

    def test(self, loader, mix_precision=False, ensemble=False, conditional_training=False):
        torch.set_grad_enabled(False)
        self.model.eval()
        running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)
        ova_len = loader.dataset._num_image
        preds_stack = None
        labels_stack = None
        running_loss = None
        for i, data in enumerate(loader):
            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            if mix_precision:
                with torch.cuda.amp.autocast():
                    preds = self.model(imgs)
                    if self.split_output:
                        preds = torch.cat([aa for aa in preds], dim=-1)
                    if conditional_training:
                        preds = preds[:, self.id_leaf]
                        labels = labels[:, self.id_leaf]
                    loss = self.metrics['loss'](preds, labels)
            else:
                preds = self.model(imgs)
                if self.split_output:
                    preds = torch.cat([aa for aa in preds], dim=-1)
                if conditional_training:
                    preds = preds[:, self.id_leaf]
                    labels = labels[:, self.id_leaf]
                loss = self.metrics['loss'](preds, labels)
            preds = nn.Sigmoid()(preds)
            if ensemble:
                preds = torch.mm(preds, self.ensemble_weights)
                labels = labels[:, self.id_obs]
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
                running_metrics[key] = tensor2numpy(self.metrics[key](
                    preds_stack, labels_stack))
        torch.set_grad_enabled(True)
        self.model.train()
        return running_metrics

    def FAEL(self, loader, test_loader, type='basic', init_lr=0.01, log_step=100, steps=20, lambda1=0.1, lambda2=2):
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
            labels = labels[:, self.id_obs]
            if type == 'basic':
                grad = (preds.T.dot(preds) + lambda1*iden_matrix).dot(w) - \
                    preds.T.dot(labels)
            elif type == 'b_constraint':
                grad = (preds.T.dot(preds) + lambda1*iden_matrix + lambda2 *
                        self.M.T.dot(self.M)).dot(w) + - preds.T.dot(labels)
            else:
                raise Exception("Not support this type!!!")
            w -= lr*grad
            if (i+1) % step == 0:
                # print(w)
                end = time.time()
                print('iter {:d} time takes: {:.3f}s'.format(i+1, end-start))
                start = time.time()
                if (i+1)//step == steps:
                    break
        self.ensemble_weights = torch.from_numpy(w).float().to(self.device)
        print('Done Essemble!!!')
        metrics = self.test(test_loader, ensemble=True)
        return metrics

    def save_FAEL_weight(self, path):
        with open(path, 'wb') as f:
            pickle.dump(tensor2numpy(self.ensemble_weights.float()), f)

    def load_FAEL_weight(self, path, mix_precision):
        with open(path, 'rb') as f:
            w = pickle.load(f)
        if mix_precision:
            self.ensemble_weights = torch.from_numpy(
                w).type(torch.HalfTensor).to(self.device)
        else:
            self.ensemble_weights = torch.from_numpy(w).float().to(self.device)
