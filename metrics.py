import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def lr_schedule(lr, lr_factor, epoch_now, lr_epochs):
    """
    Learning rate schedule with respect to epoch
    lr: float, initial learning rate
    lr_factor: float, decreasing factor every epoch_lr
    epoch_now: int, the current epoch
    lr_epochs: list of int, decreasing every epoch in lr_epochs
    return: lr, float, scheduled learning rate.
    """
    count = 0
    for epoch in lr_epochs:
        if epoch_now >= epoch:
            count += 1
            continue

        break

    return lr * np.power(lr_factor, count)

def get_loss(output, target, index, device, cfg):
    if cfg.criterion == 'BCE':
        for num_class in cfg.num_classes:
            assert num_class == 1
        target = target[:, index].view(-1)
        pos_weight = torch.from_numpy(
            np.array(cfg.pos_weight,
                    dtype=np.float32)).to(device).type_as(target)
        if cfg.batch_weight:
            if target.sum() == 0:
                loss = torch.tensor(0., requires_grad=True).to(device)
            else:
                weight = (target.size()[0] - target.sum()) / target.sum()
                loss = F.binary_cross_entropy_with_logits(
                    output[index].view(-1), target, pos_weight=weight)
        else:
            loss = F.binary_cross_entropy_with_logits(
                output[index].view(-1), target, pos_weight=pos_weight[index])

        label = torch.sigmoid(output[index].view(-1)).ge(0.5).float()
        acc = (target == label).float().sum() / len(label)
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))

    return (loss, acc)

class F1(nn.Module):
    def __init__(self, thresh=0.5, eps=1e-5):
        super(F1, self).__init__()
        self.thresh=thresh
        self.eps=eps
    def forward(self, pred, target):
        pred_thresh = (pred > self.thresh)*1.0
        tp = torch.sum((pred_thresh == target)*(target==1.0), dim=0)
        fp = torch.sum((pred_thresh != target)*(target==0.0), dim=0)
        fn = torch.sum((pred_thresh != target)*(target==1.0), dim=0)
        recall = tp / (fn+tp+self.eps)
        precision = tp / (fp+tp+self.eps)
        f1_score = 2 * precision * recall / (precision + recall + self.eps)
        return f1_score

class ACC(nn.Module):
    def __init__(self, thresh=0.5):
        super(ACC, self).__init__()
        self.thresh=thresh
    def forward(self, pred, target):
        pred_thresh = (pred > self.thresh)*1.0
        t = torch.sum(pred_thresh==target, dim=0)
        return t/pred.size()[0]