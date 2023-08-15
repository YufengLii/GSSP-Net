import torch
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss

loss_mse =  torch.nn.MSELoss(reduction='none')



def cross_entropy_loss_RCF(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label==1).float()
    num_negative = torch.sum(label==0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy(
            prediction, labelf, weight=mask, reduction='mean')

    return cost

def mseGloballoss(inputs, targets):


    targets = targets.long()
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)
    cost = loss_mse(inputs, targets.float())
    cost = cost*mask
    cost = torch.sum(cost.float().mean((1, 2, 3)))

    return cost

def weighted_l1_loss_off(logits, target, mask=None):
    loss = F.l1_loss(logits, target, reduction='none')
    if mask is not None:
        w = mask.mean(3, True).mean(2, True)
        w[w == 0] = 1
        loss = loss * (mask / w)
    return loss.mean()

def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    # cost = torch.mean(cost.float().mean((1, 2, 3))) # before sum
    cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
    return l_weight*cost

def weighted_l1_loss(logits, target, mask=None):
    loss = F.l1_loss(logits, target, reduction='sum')
    # loss = F.binary_cross_entropy(logits, target)

    if mask is not None:
        w = mask.mean(3, True).mean(2, True)
        w[w == 0] = 1
        loss = loss * (mask / w)
    # return loss.mean()
    return loss


def focal_loss_batch(line_label, link_preds):
    loss_link = [
        sigmoid_focal_loss(pred_label,
                           gt_lable,
                           alpha=0.25,
                           gamma=2.0,
                           reduction='mean')
        for gt_lable, pred_label in zip(line_label, link_preds)
    ]
    return sum(loss_link) / 8.0


def loss_link_binary_cross_entropy(link_preds, line_label):

    loss_link = [
        F.binary_cross_entropy(torch.sigmoid(pred),
                               lable,
                               reduction='mean')
        for lable, pred in zip(line_label, link_preds)
    ]
    # loss_link = loss_link[0]+loss_link[1]+loss_link[2]+loss_link[3]+loss_link[4]+loss_link[5]+loss_link[6]+loss_link[7]
    return sum(loss_link) / 8.0


def hed_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.1).float()).float()
    num_negative = torch.sum((mask <= 0.).float()).float()

    mask[mask > 0.1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)
    inputs = torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='sum')(inputs.float(),
                                                   targets.float())

    # return l_weight*torch.sum(cost)
    return l_weight * cost


def bdcn_loss2(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    # cost = torch.mean(cost.float().mean((1, 2, 3))) # before sum
    cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
    return l_weight*cost


def bdcn_loss3(inputs, targets, l_weight=1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    # targets[targets>0.05] = 1
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.1).float()).float()  # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float()  # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative)  # 0.1
    mask[mask <= 0.] = 1.1 * num_positive / \
        (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    # inputs = torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())

    cost = cost.float().mean()
    # cost = torch.mean(cost.float().mean((1, 2, 3))) # before sum
    # cost = torch.sum(cost.float().mean((1, 2, 3)))  # before sum
    # return l_weight*cost
    return l_weight * cost
