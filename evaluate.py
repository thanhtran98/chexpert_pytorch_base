import torch
import numpy as np
from train import get_loss
from sklearn import metrics

def evaluate(cfg, device, model, dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)

    predlist = list(x for x in range(len(cfg.num_classes)))
    true_list = list(x for x in range(len(cfg.num_classes)))
    for step in range(steps):
        image, target = next(dataiter)
        image = image.to(device)
        target = target.to(device)
        output = model(image)
        auclist = []
        # different number of tasks
        for t in range(len(cfg.num_classes)):

            loss_t, acc_t = get_loss(output, target, t, device, cfg)
            # AUC
            output_tensor = torch.sigmoid(
                output[t].view(-1)).cpu().detach().numpy()
            target_tensor = target[:, t].view(-1).cpu().detach().numpy()
            if step == 0:
                predlist[t] = output_tensor
                true_list[t] = target_tensor
            else:
                predlist[t] = np.append(predlist[t], output_tensor)
                true_list[t] = np.append(true_list[t], target_tensor)

            fpr, tpr, thresholds = metrics.roc_curve(
                    true_list[t], predlist[t], pos_label=1)
            auc = metrics.auc(fpr, tpr)
            auclist.append(auc)

            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()

        auclist = np.array(auclist)

    return acc_sum/steps, loss_sum/steps, auclist