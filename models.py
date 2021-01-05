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
                    # nn.Conv2d(self.num_features,num_class,kernel_size=1,
                    #           stride=1,padding=0,bias=True)
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
            # (N, 1, H, W)
            # logit_map = None
            # if not (self.cfg.global_pool == 'AVG_MAX' or
            #         self.cfg.global_pool == 'AVG_MAX_LSE'):
            #     logit_map = classifier(feat_map)
            #     logit_maps.append(logit_map.squeeze())
            # (N, C, 1, 1)
            # feat = self.global_pool(feat_map, logit_map)
            feat = self.avgpool(feat_map)

            # if self.cfg.fc_bn:
            #     bn = getattr(self, "bn_" + str(index))
            #     feat = bn(feat)
            feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)
            # (N, num_class, 1, 1)

            logit = classifier(feat)

            # (N, num_class)
            logit = logit
            logits.append(logit)

        return logits