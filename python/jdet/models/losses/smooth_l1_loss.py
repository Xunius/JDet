import jittor as jt 
from jdet.utils.registry import LOSSES
from jittor import nn

def smooth_l1_loss(pred,target,weight=None,beta=1.,avg_factor=None,reduction="mean"):
    diff = jt.abs(pred-target)
    if beta!=0.:
        flag = (diff<beta).float()
        loss = flag*0.5* diff.sqr()/beta + (1-flag)*(diff - 0.5 * beta)
    else:
        loss = diff 

    if weight is not None:
        if weight.ndim==1:
            weight = weight[:,None]
        loss *= weight

    if avg_factor is None:
        avg_factor = max(loss.shape[0],1)

    if reduction == "mean":
        loss = loss.sum()/avg_factor
    elif reduction == "sum":
        loss = loss.sum()

    return loss 

def smooth_l1_loss2(pred,target,weight=None,beta=1.,avg_factor=None,reduction="mean"):

    weight = weight.mean(dim=1)
    idx = weight > 0
    if idx.sum() == 0:
        loss = pred.mean() * 0.
        return loss

    pred = pred[idx, :]
    target = target[idx, :]

    scale_weight = (jt.maximum(pred[:, 2:4].abs(), target[:, 2:4].abs())).sqr()

    center_loss = (pred[:, 0:2] - target[:, 0:2]).sqr()
    cidx = center_loss > scale_weight
    center_loss[cidx] = (center_loss / (scale_weight + 1e-5))[cidx]

    wh_loss = pred[:, 2:4] - target[:, 2:4]
    wh_loss = wh_loss.sqr() / (scale_weight + 1e-5)

    angle_loss = pred[:, -1] - target[:, -1]
    angle_loss = jt.cos(angle_loss).sqr() * center_loss[:, 0] + jt.sin(angle_loss).sqr() * center_loss[:, 1]

    loss = center_loss.sum(dim=1) + wh_loss.sum(dim=1) + angle_loss

    #if weight is not None:
        #if weight.ndim==1:
            #weight = weight[:,None]
        #loss *= weight
    #loss *= weight

    if avg_factor is None:
        avg_factor = max(loss.shape[0],1)

    if reduction == "mean":
        loss = loss.sum()/max(avg_factor,1)
    elif reduction == "sum":
        loss = loss.sum()

    return loss 

def rotated_box_to_poly(rrects):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    n = rrects.shape[0]
    if n == 0:
        return jt.zeros([0,8])
    x_ctr = rrects[:, 0] 
    y_ctr = rrects[:, 1] 
    width = rrects[:, 2] 
    height = rrects[:, 3] 
    angle = rrects[:, 4]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    rect = jt.stack([tl_x, br_x, br_x, tl_x, tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y, tl_y, tl_y, br_y, br_y], 1).reshape([n, 2, 8])
    c = jt.cos(angle)
    s = jt.sin(angle)
    R = jt.stack([c, c, c, c, s, s, s, s, -s, -s, -s, -s, c, c, c, c], 1).reshape([n, 2, 8])
    offset = jt.stack([x_ctr, x_ctr, x_ctr, x_ctr, y_ctr, y_ctr, y_ctr, y_ctr], 1)
    poly = ((R * rect).sum(1) + offset).reshape([n, 2, 4]).permute([0,2,1]).reshape([n, 8])
    return poly


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
        #loss_bbox = self.loss_weight * smooth_l1_loss3(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_bbox
