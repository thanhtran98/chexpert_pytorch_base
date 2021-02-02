import torch.nn as nn
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from model.models import ResNeSt_parallel, Efficient_parallel, Efficient, ResNeSt, Dense, Dense_parallel
from resnest.torch import resnest50, resnest101, resnest200, resnest269
from efficientnet_pytorch import EfficientNet
from torchvision.models import densenet121, densenet161, densenet169, densenet201


def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
    if norm_type == 'BatchNorm':
        return nn.BatchNorm2d(num_features, eps=eps)
    elif norm_type == "GroupNorm":
        return nn.GroupNorm(num_groups, num_features, eps=eps)
    elif norm_type == "InstanceNorm":
        return nn.InstanceNorm2d(num_features, eps=eps,
                                 affine=True, track_running_stats=True)
    else:
        raise Exception('Unknown Norm Function : {}'.format(norm_type))


def get_optimizer(params, cfg):
    if cfg.optimizer == 'SGD':
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adadelta':
        return Adadelta(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adagrad':
        return Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'RMSprop':
        return RMSprop(params, lr=cfg.lr, momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(cfg.optimizer))

def get_model(cfg):
    if cfg.backbone == 'resnest':
        childs_cut = 9
        if cfg.id == '50':
            pre_name = resnest50
        elif cfg.id == '101':
            pre_name = resnest101
        elif cfg.id == '200':
            pre_name = resnest200
        else:
            pre_name = resnest269
        pre_model = pre_name(pretrained=cfg.pretrained)
        for param in pre_model.parameters():
            param.requires_grad = True
        if cfg.split_output:
            model = ResNeSt(pre_model, cfg)
        else:
            model = ResNeSt_parallel(pre_model, 5)
    elif cfg.backbone == 'efficient' or cfg.backbone == 'efficientnet':
        childs_cut = 6
        pre_name = 'efficientnet-'+cfg.id
        if cfg.pretrained:
            pre_model = EfficientNet.from_pretrained(pre_name)
        else:
            pre_model = EfficientNet.from_name(pre_name)
        for param in pre_model.parameters():
            param.requires_grad = True
        if cfg.split_output:
            model = Efficient(pre_model, cfg)
        else:
            model = Efficient_parallel(pre_model, 5)
    elif cfg.backbone == 'dense' or cfg.backbone == 'densenet':
        childs_cut = 2
        if cfg.id == '121':
            pre_name = densenet121
        elif cfg.id == '161':
            pre_name = densenet161
        elif cfg.id == '169':
            pre_name = densenet169
        else:
            pre_name = densenet201
        pre_model = pre_name(pretrained=cfg.pretrained)
        for param in pre_model.parameters():
            param.requires_grad = True
        if cfg.split_output:
            model = Dense(pre_model, cfg)
        else:
            model = Dense_parallel(pre_model, 5)
    else:
        raise Exception("Not support this model!!!!")
    return model, childs_cut

def get_str(metrics, mode, s):
    for key in list(metrics.keys()):
        if key == 'loss':
            s += "{}_{} {:.3f} - ".format(mode, key, metrics[key])
        else:
            metric_str = ' '.join(
                map(lambda x: '{:.5f}'.format(x), metrics[key]))
            s += "{}_{} {} - ".format(mode, key, metric_str)
    s = s[:-2] + '\n'
    return s

def tensor2numpy(input_tensor):
    # device cuda Tensor to host numpy
    return input_tensor.cpu().detach().numpy()
