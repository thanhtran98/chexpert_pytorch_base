import torch.nn as nn
import torch
import torch.nn.functional as F
from model.global_pool import GlobalPool


class ResNeSt(nn.Module):
    def __init__(self, pre_model, cfg):
        """ResNeSt-based model - split the head after the model backbone into branches

        Args:
            pre_model (torch.nn.module): predenfined ResNeSt backbone model.
            cfg (dict): configuration file.
        """
        super(ResNeSt, self).__init__()
        self.cfg = cfg
        self.conv1 = pre_model.conv1
        self.bn1 = pre_model.bn1
        self.relu = pre_model.relu
        self.maxpool = pre_model.maxpool
        self.layer1 = pre_model.layer1
        self.layer2 = pre_model.layer2
        self.layer3 = pre_model.layer3
        self.layer4 = pre_model.layer4
        if self.cfg.modify_gp:
            self.avgpool = GlobalPool(self.cfg)
        else:
            self.avgpool = pre_model.avgpool
        self.num_features = pre_model.fc.in_features
        self._init_classifier()

    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.modify_gp:
                setattr(self, "fc_"+str(index),
                        nn.Conv2d(self.num_features, num_class, kernel_size=1,
                                  stride=1, padding=0, bias=True)
                        )
            else:
                setattr(self, "fc_"+str(index),
                        nn.Linear(in_features=self.num_features,
                                  out_features=num_class, bias=True)
                        )

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def forward(self, x):
        # (N, C, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        for index, num_class in enumerate(self.cfg.num_classes):

            classifier = getattr(self, "fc_" + str(index))

            if self.cfg.modify_gp:
                logit_map = None
                if not (self.cfg.global_pool == 'AVG_MAX' or
                        self.cfg.global_pool == 'AVG_MAX_LSE'):
                    logit_map = classifier(feat_map)
                    logit_maps.append(logit_map.squeeze())
                feat = self.avgpool(feat_map, logit_map)

                if self.cfg.fc_bn:
                    bn = getattr(self, "bn_" + str(index))
                    feat = bn(feat)
                feat = F.dropout(feat, p=self.cfg.fc_drop,
                                 training=self.training)
                # (N, num_class, 1, 1)

                logit = classifier(feat)
                logit = logit.squeeze(-1).squeeze(-1)

            else:
                feat = self.avgpool(feat_map)

                feat = F.dropout(feat, p=self.cfg.fc_drop,
                                 training=self.training)

                # (N, num_class, 1, 1)
                logit = classifier(feat)

                # (N, num_class)

            logits.append(logit)

        return logits


class Dense(nn.Module):
    def __init__(self, pre_model, cfg):
        """DenseNet-based model - split the head after the model backbone into branches

        Args:
            pre_model (torch.nn.module): predenfined DenseNet backbone model.
            cfg (dict): configuration file.
        """
        super(Dense, self).__init__()
        self.cfg = cfg
        self.features = pre_model.features
        if self.cfg.modify_gp:
            self.avgpool = GlobalPool(self.cfg)
        else:
            self.avgpool = F.adaptive_avg_pool2d
        self.num_features = pre_model.classifier.in_features
        self._init_classifier()

    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.modify_gp:
                setattr(self,"fc_"+str(index),
                    nn.Conv2d(self.num_features,num_class,kernel_size=1,
                              stride=1,padding=0,bias=True)
                    )
            else:
                setattr(self,"fc_"+str(index),
                        nn.Linear(in_features=self.num_features,
                                out_features=num_class, bias=True)
                        )

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def forward(self, x):
        # (N, C, H, W)
        x = self.features(x)
        feat_map = F.relu(x, inplace=True)
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        for index, num_class in enumerate(self.cfg.num_classes):

            classifier = getattr(self, "fc_" + str(index))

            if self.cfg.modify_gp:
                logit_map = None
                if not (self.cfg.global_pool == 'AVG_MAX' or
                        self.cfg.global_pool == 'AVG_MAX_LSE'):
                    logit_map = classifier(feat_map)
                    logit_maps.append(logit_map.squeeze())
                feat = self.avgpool(feat_map, logit_map)

                if self.cfg.fc_bn:
                    bn = getattr(self, "bn_" + str(index))
                    feat = bn(feat)
                feat = F.dropout(feat, p=self.cfg.fc_drop,
                                 training=self.training)
                # (N, num_class, 1, 1)

                logit = classifier(feat)
                logit = logit.squeeze(-1).squeeze(-1)

            else:
                # logit_map = classifier(feat_map)
                # logit_maps.append(logit_map.squeeze())
                feat = self.avgpool(feat_map, (1, 1))
                feat = torch.flatten(feat, 1)
                # feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)

                # (N, num_class, 1, 1)
                logit = classifier(feat)

                # (N, num_class)

            logits.append(logit)

        return logits


class Efficient(nn.Module):
    def __init__(self, pre_model, cfg, heatmap=False):
        """EfficientNet-based model - split the head after the model backbone into branches

        Args:
            pre_model (torch.nn.module): predenfined EfficientNet backbone model.
            cfg (dict): configuration file.
            heatmap (bool, optional): output heatmap for visualization. Defaults to False.
        """
        super(Efficient, self).__init__()
        self.cfg = cfg
        self.heatmap = heatmap
        self._conv_stem = pre_model._conv_stem
        self._bn0 = pre_model._bn0
        self._blocks = pre_model._blocks
        self._conv_head = pre_model._conv_head
        self._bn1 = pre_model._bn1
        if self.cfg.modify_gp:
            self._avg_pooling = GlobalPool(self.cfg)
        else:
            self._avg_pooling = pre_model._avg_pooling
        self._dropout = pre_model._dropout
        self.num_features = pre_model._fc.in_features
        self._init_classifier()

    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            setattr(self, "fc_"+str(index),
                    nn.Conv2d(self.num_features, num_class, kernel_size=1,
                              stride=1, padding=0, bias=True)
                    )

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def forward(self, x):
        # (N, C, H, W)
        x = self._conv_stem(x)
        x = self._bn0(x)
        for block in self._blocks:
            x = block(x)
        x = self._conv_head(x)
        feat_map = self._bn1(x)
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        for index, num_class in enumerate(self.cfg.num_classes):
            
            classifier = getattr(self, "fc_" + str(index))

            if self.cfg.modify_gp:
                logit_map = None
                if not (self.cfg.global_pool == 'AVG_MAX' or
                        self.cfg.global_pool == 'AVG_MAX_LSE'):
                    logit_map = classifier(feat_map)
                    logit_maps.append(logit_map.squeeze())
                feat = self._avg_pooling(feat_map, logit_map)

                if self.cfg.fc_bn:
                    bn = getattr(self, "bn_" + str(index))
                    feat = bn(feat)
                feat = F.dropout(feat, p=self.cfg.fc_drop,
                                 training=self.training)
                # (N, num_class, 1, 1)

                logit = classifier(feat)
                logit = logit.squeeze(-1).squeeze(-1)

            else:
                logit_map = classifier(feat_map)
                logit_maps.append(logit_map.squeeze())
                # (N, C, 1, 1)
                feat = self._avg_pooling(feat_map)
                feat = self._dropout(feat)

                # (N, num_class, 1, 1)
                # feat = feat.squeeze(-1).squeeze(-1)
                logit = classifier(feat)
                # logit = self._swish(logit)

                # (N, num_class)
                logit = logit.squeeze(-1).squeeze(-1)

            logits.append(logit)

        if self.heatmap:
            return logits, logit_maps
        else:
            return logits


class ResNeSt_parallel(nn.Module):
    def __init__(self, pre_model, num_classes):
        """ResNeSt-based model - not split the head of the model, instead use one linear layer to return directly outputs

        Args:
            pre_model (torch.nn.module): predenfined ResNeSt backbone model.
            num_classes (int): number of classes.
        """
        super(ResNeSt_parallel, self).__init__()
        self.conv1 = pre_model.conv1
        self.bn1 = pre_model.bn1
        self.relu = pre_model.relu
        self.maxpool = pre_model.maxpool
        self.layer1 = pre_model.layer1
        self.layer2 = pre_model.layer2
        self.layer3 = pre_model.layer3
        self.layer4 = pre_model.layer4
        self.avgpool = pre_model.avgpool
        self.num_features = pre_model.fc.in_features
        self.fc = nn.Linear(in_features=self.num_features,
                            out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class Dense_parallel(nn.Module):
    def __init__(self, pre_model, num_classes):
        """DenseNet-based model - not split the head of the model, instead use one linear layer to return directly outputs

        Args:
            pre_model (torch.nn.module): predenfined DenseNet backbone model.
            num_classes (int): number of classes.
        """
        super(Dense_parallel, self).__init__()
        self.features = pre_model.features
        self.num_features = pre_model.classifier.in_features
        self.fc = nn.Linear(in_features=self.num_features,
                            out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        out = F.relu(x, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class Efficient_parallel(nn.Module):
    def __init__(self, pre_model, num_classes):
        """EfficientNet-based model - not split the head of the model, instead use one linear layer to return directly outputs

        Args:
            pre_model (torch.nn.module): predenfined EfficientNet backbone model.
            num_classes (int): number of classes.
        """
        super(Efficient_parallel, self).__init__()
        self._conv_stem = pre_model._conv_stem
        self._bn0 = pre_model._bn0
        self._blocks = pre_model._blocks
        self._conv_head = pre_model._conv_head
        self._bn1 = pre_model._bn1
        self._avg_pooling = pre_model._avg_pooling
        self._dropout = pre_model._dropout
        self.num_features = pre_model._fc.in_features
        self._fc = nn.Linear(in_features=self.num_features,
                             out_features=num_classes, bias=True)

    def forward(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        for block in self._blocks:
            x = block(x)
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._avg_pooling(x)
        x = self._dropout(x)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        x = self._fc(x)
        return x