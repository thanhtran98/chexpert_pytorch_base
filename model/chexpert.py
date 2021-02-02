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
from model.utils import get_model, get_str, tensor2numpy, get_optimizer

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

    def __init__(self, cfg, loss_func, metrics=None):
        """CheXpert class contains all functions used for training and testing our models

        Args:
            cfg (dict): configuration file.
            loss_func (torch.nn.Module): loss function of the model.
            metrics (dict, optional): metrics use to evaluate model performance. Defaults to None.
        """
        self.cfg = cfg
        if self.cfg.full_classes:
            self.cfg.num_classes = 14*[1]
        self.model, self.childs_cut = get_model(self.cfg)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_func = loss_func
        if metrics is not None:
            self.metrics = metrics
            self.metrics['loss'] = self.loss_func
        else:
            self.metrics = {'loss': self.loss_func}
        self.optimizer = get_optimizer(self.model.parameters(), self.cfg)
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

    def save_ckp(self, ckp_path, epoch, iter):
        """Save checkpoint

        Args:
            ckp_path (str): path to saved checkpoint
            epoch (int): current epoch
            iter (int): current iteration
        """
        if os.path.exists(os.path.dirname(ckp_path)):
            torch.save(
                {'epoch': epoch+1,
                 'iter': iter+1,
                 'state_dict': self. model.state_dict()},
                ckp_path
            )
        else:
            print("Save path not exist!!!")

    def predict(self, image):
        """Run prediction

        Args:
            image (torch.Tensor): images to predict. Shape (batch size, C, H, W)

        Returns:
            torch.Tensor: model prediction
        """
        torch.set_grad_enabled(False)
        self.model.eval()
        with torch.no_grad() as tng:
            if self.cfg.mix_precision:
                with torch.cuda.amp.autocast():
                    preds = self.model(image)
                    if self.cfg.split_output:
                        preds = torch.cat([aa for aa in preds], dim=-1)
                    if self.cfg.conditional_training:
                        preds = preds[:, self.id_leaf]
            else:
                preds = self.model(image)
                if self.cfg.split_output:
                    preds = torch.cat([aa for aa in preds], dim=-1)
                if self.cfg.conditional_training:
                    preds = preds[:, self.id_leaf]

        return preds

    def predict_from_file(self, image_file):
        """Run prediction from image path

        Args:
            image_file (str): image path

        Returns:
            numpy array: model prediction in numpy array type
        """
        image_gray = cv2.imread(image_file, 0)
        image = transform(image_gray, self.cfg)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)

        return tensor2numpy(nn.Sigmoid()(self.predict(image)))

    def predict_loader(self, loader, ensemble=False):
        """Run prediction on a given dataloader.

        Args:
            loader (torch.utils.data.Dataloader): a dataloader
            ensemble (bool, optional): use FAEL for prediction. Defaults to False.

        Returns:
            torch.Tensor, torch.Tensor: prediction, labels
        """
        preds_stack = None
        labels_stack = None

        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            preds = self.predict(imgs, self.cfg.mix_precision, self.cfg.conditional_training)
            if self.cfg.conditional_training:
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

    def train(self, train_loader, val_loader, epochs=10, iter_log=100, use_lr_sch=False, resume=False, ckp_dir='./experiment/checkpoint',
              writer=None, eval_metric='loss'):
        """Run training

        Args:
            train_loader (torch.utils.data.Dataloader): dataloader use for training
            val_loader (torch.utils.data.Dataloader): dataloader use for validation
            epochs (int, optional): number of training epochs. Defaults to 120.
            iter_log (int, optional): logging iteration. Defaults to 100.
            use_lr_sch (bool, optional): use learning rate scheduler. Defaults to False.
            resume (bool, optional): resume training process. Defaults to False.
            ckp_dir (str, optional): path to checkpoint directory. Defaults to './experiment/checkpoint'.
            writer (torch.utils.tensorboard.SummaryWriter, optional): tensorboard summery writer. Defaults to None.
            eval_metric (str, optional): name of metric for validation. Defaults to 'loss'.
        """
        if use_lr_sch:
            lr_sch = torch.optim.lr_scheduler.StepLR(
                self.optimizer, int(epochs*2/3), self.cfg.lr/3)
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
        if self.cfg.mix_precision:
            print('Train with mix precision!')
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(epoch_resume-1, epochs):
            start = time.time()
            running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)
            running_metrics.pop(eval_metric, None)
            n_iter = len(train_loader)
            torch.set_grad_enabled(True)
            self.model.train()
            batch_weights = (1/iter_log)*np.ones(n_iter)
            step_per_epoch = n_iter // iter_log
            if n_iter % iter_log:
                step_per_epoch += 1
                batch_weights[-(n_iter % iter_log):] = 1 / (n_iter % iter_log)
                iter_per_step = iter_log*np.ones(step_per_epoch, dtype = np.int16)
                iter_per_step[-1] = n_iter%iter_log
            else:
                iter_per_step = iter_log*np.ones(step_per_epoch, dtype = np.int16)
            i = 0
            for step in range(step_per_epoch):
                for iteration in tqdm.tqdm(range(iter_per_step[step])):
                    data = next(iter(train_loader))
                    imgs, labels = data[0].to(self.device), data[1].to(self.device)
                    if self.cfg.mix_precision:
                        with torch.cuda.amp.autocast():
                            preds = self.model(imgs)
                            if self.cfg.split_output:
                                preds = torch.cat([aa for aa in preds], dim=-1)
                            if self.cfg.conditional_training:
                                preds = preds[:, self.id_leaf]
                                labels = labels[:, self.id_leaf]
                            loss = self.metrics['loss'](preds, labels)
                    else:
                        preds = self.model(imgs)
                        if self.cfg.split_output:
                            preds = torch.cat([aa for aa in preds], dim=-1)
                        if self.cfg.conditional_training:
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
                    if self.cfg.mix_precision:
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    i += 1
                
                s = "Epoch [{}/{}] Iter [{}/{}]:\n".format(
                    epoch+1, epochs, i+1, n_iter)
                s = get_str(running_metrics, 'train', s)
                running_metrics_test = self.test(
                    val_loader, False)
                s = get_str(running_metrics_test, 'val', s)
                if self.cfg.conditional_training or (not self.cfg.full_classes):
                    metric_eval = running_metrics_test[eval_metric]
                else:
                    metric_eval = running_metrics_test[eval_metric][self.id_obs]
                s = s[:-1] + "- mean_"+eval_metric + \
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
                s += " ({:.1f}s)".format(end-start)
                print(s)
                if metric_eval.mean() > best_metric:
                    best_metric = metric_eval.mean()
                    shutil.copyfile(os.path.join(ckp_dir, 'latest.ckpt'), os.path.join(
                        ckp_dir, 'epoch'+str(epoch+1)+'_iter'+str(i+1)+'.ckpt'))
                    print('new checkpoint saved!')
                start = time.time()

            s = "Epoch [{}/{}] Iter [{}/{}]:\n".format(
                epoch+1, epochs, n_iter, n_iter)
            running_metrics = self.test(val_loader)
            s = get_str(running_metrics, 'val', s)
            if self.cfg.conditional_training or (not self.cfg.full_classes):
                metric_eval = running_metrics[eval_metric]
            else:
                metric_eval = running_metrics[eval_metric][self.id_obs]
            s = s[:-1] + "- mean_"+eval_metric + \
                " {:.3f}".format(metric_eval.mean())
            end = time.time()
            s += " ({:.1f}s)".format(end-start)
            print(s)
            if lr_sch is not None:
                lr_sch.step()
                print('current lr: {:.4f}'.format(lr_sch.get_lr()[0]))

    def test(self, loader, ensemble=False):
        """Run testing

        Args:
            loader (torch.utils.data.Dataloader): dataloader use for testing
            ensemble (bool, optional): use FAEL for prediction. Defaults to False.

        Returns:
            dict: metrics use to evaluate model performance.
        """
        torch.set_grad_enabled(False)
        self.model.eval()
        running_metrics = dict.fromkeys(self.metrics.keys(), 0.0)
        ova_len = loader.dataset._num_image
        preds_stack = None
        labels_stack = None
        running_loss = None
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, labels = data[0].to(self.device), data[1].to(self.device)
            if self.cfg.mix_precision:
                with torch.cuda.amp.autocast():
                    preds = self.model(imgs)
                    if self.cfg.split_output:
                        preds = torch.cat([aa for aa in preds], dim=-1)
                    if self.cfg.conditional_training:
                        preds = preds[:, self.id_leaf]
                        labels = labels[:, self.id_leaf]
                    loss = self.metrics['loss'](preds, labels)
            else:
                preds = self.model(imgs)
                if self.cfg.split_output:
                    preds = torch.cat([aa for aa in preds], dim=-1)
                if self.cfg.conditional_training:
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

    def FAEL(self, loader, val_loader, type='basic', init_lr=0.01, log_iter=100, steps=20, lambda1=0.1, lambda2=2):
        """Run fully associative ensemble learning (FAEL)

        Args:
            loader (torch.utils.data.Dataloader): dataloader use for training FAEL model
            val_loader (torch.utils.data.Dataloader): dataloader use for validating FAEL model
            type (str, optional): regularization type (basic/binary constraint). Defaults to 'basic'.
            init_lr (float, optional): initial learning rate. Defaults to 0.01.
            log_step (int, optional): logging step. Defaults to 100.
            steps (int, optional): total steps. Defaults to 20.
            lambda1 (float, optional): l2 regularization parameter. Defaults to 0.1.
            lambda2 (int, optional): binary constraint parameter. Defaults to 2.

        Returns:
            dict: metrics use to evaluate model performance.
        """
        n_class = len(self.cfg.num_classes)
        w = np.random.rand(n_class, len(self.id_obs))
        iden_matrix = np.diag(np.ones(n_class))
        lr = init_lr
        start = time.time()
        for i, data in enumerate(tqdm.tqdm(loader, total=log_iter*steps)):
            imgs, labels = data[0].to(self.device), data[1]
            preds = self.predict(imgs)
            preds = tensor2numpy(preds)
            labels = tensor2numpy(labels)
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
            if (i+1) % log_iter == 0:
                # print(w)
                end = time.time()
                print('iter {:d} time takes: {:.3f}s'.format(i+1, end-start))
                start = time.time()
                if (i+1)//log_iter == steps:
                    break
        if self.cfg.mix_precision:
            self.ensemble_weights = torch.from_numpy(
                w).type(torch.HalfTensor).to(self.device)
        else:
            self.ensemble_weights = torch.from_numpy(w).float().to(self.device)
        print('Done Essemble!!!')
        metrics = self.test(val_loader, ensemble=True)

        return metrics

    def save_FAEL_weight(self, path):
        """save FAEL weight

        Args:
            path (str): path to saved weight.
        """
        with open(path, 'wb') as f:
            pickle.dump(tensor2numpy(self.ensemble_weights.float()), f)

    def load_FAEL_weight(self, path):
        """load FAEL weight

        Args:
            path (str): path to saved weight
            mix_precision (bool, optional): use mix precision for prediction. Defaults to False.
        """
        with open(path, 'rb') as f:
            w = pickle.load(f)
        if self.cfg.mix_precision:
            self.ensemble_weights = torch.from_numpy(
                w).type(torch.HalfTensor).to(self.device)
        else:
            self.ensemble_weights = torch.from_numpy(w).float().to(self.device)
