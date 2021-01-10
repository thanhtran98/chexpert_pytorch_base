import torch.nn as nn
import torch.nn.functional as F

class ResNeSt(nn.Module):
    def __init__(self, pre_model, cfg):
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
        self.avgpool = pre_model.avgpool
        self.num_features = pre_model.fc.in_features
        self._init_classifier()

    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
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
            if self.cfg.attention_map != "None":
                feat_map = self.attention_map(feat_map)

            classifier = getattr(self, "fc_" + str(index))
            
            feat = self.avgpool(feat_map)

            feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)
            
            # (N, num_class, 1, 1)
            logit = classifier(feat)

            # (N, num_class)
            logits.append(logit)

        return logits


class Efficient(nn.Module):
    def __init__(self, pre_model, cfg):
        super(Efficient, self).__init__()
        self.cfg = cfg
        self._conv_stem = pre_model._conv_stem
        self._bn0 = pre_model._bn0
        self._blocks = pre_model._blocks
        self._conv_head = pre_model._conv_head
        self._bn1 = pre_model._bn1
        self._avg_pooling = pre_model._avg_pooling
        self._dropout = pre_model._dropout
        self._swish = pre_model._swish
        self.num_features = pre_model._fc.in_features
        self._init_classifier()

    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            setattr(self,"fc_"+str(index),
                    nn.Conv2d(self.num_features,num_class,kernel_size=1,
                              stride=1,padding=0,bias=True)
                    # nn.Linear(in_features=self.num_features,
                    #           out_features=num_class, bias=True)
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
            if self.cfg.attention_map != "None":
                feat_map = self.attention_map(feat_map)

            classifier = getattr(self, "fc_" + str(index))

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

        return logits