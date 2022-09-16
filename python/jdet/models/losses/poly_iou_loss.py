import numpy as np
import jittor as jt 
from jittor import nn 

from jdet.utils.registry import LOSSES


from jdet.ops.convex_sort import convex_sort
from jdet.ops.bbox_transforms import bbox2type, get_bbox_areas
from jdet.models.boxes.iou_calculator import bbox_overlaps_rotated
from jdet.models.boxes.box_ops import rotated_box_to_poly


def shoelace(pts):
    roll_pts = jt.roll(pts, 1, dims=-2)
    xyxy = pts[..., 0] * roll_pts[..., 1] - \
           roll_pts[..., 0] * pts[..., 1]
    areas = 0.5 * jt.abs(xyxy.sum(dim=-1))
    return areas


def convex_areas(pts, masks):
    nbs, npts, _ = pts.size()
    index = convex_sort(pts, masks)
    index[index == -1] = npts
    index = index[..., None].repeat(1, 1, 2)

    ext_zeros = jt.zeros((nbs, 1, 2),dtype=pts.dtype)
    ext_pts = jt.concat([pts, ext_zeros], dim=1)
    polys = jt.gather(ext_pts, 1, index)

    xyxy_1 = (polys[:, 0:-1, 0] * polys[:, 1:, 1])
    xyxy_2 = (polys[:, 0:-1, 1] * polys[:, 1:, 0])
    xyxy_1.sync()
    xyxy_2.sync()

    xyxy = xyxy_1 - xyxy_2
    areas = 0.5 * jt.abs(xyxy.sum(dim=-1))
    return areas


def poly_intersection(pts1, pts2, areas1=None, areas2=None, eps=1e-6):
    # Calculate the intersection points and the mask of whether points is inside the lines.
    # Reference:
    #    https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    #    https://github.com/lilanxiao/Rotated_IoU/blob/master/box_intersection_2d.py
    lines1 = jt.concat([pts1, jt.roll(pts1, -1, dims=1)], dim=2)
    lines2 = jt.concat([pts2, jt.roll(pts2, -1, dims=1)], dim=2)
    lines1, lines2 = lines1.unsqueeze(2), lines2.unsqueeze(1)
    x1, y1, x2, y2 = lines1.unbind(dim=-1) # dim: N, 4, 1
    x3, y3, x4, y4 = lines2.unbind(dim=-1) # dim: N, 1, 4

    num = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    with jt.no_grad():
        den_u = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
        t, u = den_t / num, den_u / num
        mask_t = (t > 0) & (t < 1)
        mask_u = (u > 0) & (u < 1)
        mask_inter = jt.logical_and(mask_t, mask_u)

    t = den_t / (num + eps)
    x_inter = x1 + t * (x2 - x1)
    y_inter = y1 + t * (y2 - y1)
    pts_inter = jt.stack([x_inter, y_inter], dim=-1)

    B = pts1.size(0)
    pts_inter = pts_inter.view(B, -1, 2)
    mask_inter = mask_inter.view(B, -1)

    # Judge if one polygon's vertices are inside another polygon.
    # Use
    with jt.no_grad():
        areas1 = shoelace(pts1) if areas1 is None else areas1
        areas2 = shoelace(pts2) if areas2 is None else areas2

        triangle_areas1 = 0.5 * jt.abs(
            (x3 - x1) * (y4 - y1) - (y3 - y1) * (x4 - x1))
        sum_areas1 = triangle_areas1.sum(dim=-1)
        mask_inside1 = jt.abs(sum_areas1 - areas2[..., None]) < 1e-3 * areas2[..., None]

        triangle_areas2 = 0.5 * jt.abs(
            (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))
        sum_areas2 = triangle_areas2.sum(dim=-2)
        mask_inside2 = jt.abs(sum_areas2 - areas1[..., None]) < 1e-3 * areas1[..., None]

    all_pts = jt.concat([pts_inter, pts1, pts2], dim=1)
    masks = jt.concat([mask_inter, mask_inside1, mask_inside2], dim=1)
    return all_pts, masks


def poly_enclose(pts1, pts2):
    all_pts = jt.concat([pts1, pts2], dim=1)
    mask1 = pts1.new_ones((pts1.size(0), pts1.size(1)))
    mask2 = pts2.new_ones((pts2.size(0), pts2.size(1)))
    masks = jt.concat([mask1, mask2], dim=1)

    return all_pts, masks

def poly_enclose_diag(pts1, pts2):

    pts1 = pts1.reshape(-1, 8)
    pts2 = pts2.reshape(-1, 8)
    all_pts = jt.concat([pts1, pts2], dim=1)
    xmin = all_pts[:, ::2].min(1)
    ymin = all_pts[:, 1::2].min(1)
    xmax = all_pts[:, ::2].max(1)
    ymax = all_pts[:, 1::2].max(1)
    diag = jt.sqrt((xmax - xmin).sqr() + (ymax - ymin).sqr())
    return diag



def poly_iou_loss(pred, target, linear=False, eps=1e-6,weight=None, reduction='mean', avg_factor=None):
    areas1, areas2 = get_bbox_areas(pred), get_bbox_areas(target)
    pred, target = bbox2type(pred, 'poly'), bbox2type(target, 'poly')

    pred_pts = pred.view(pred.size(0), -1, 2)
    target_pts = target.view(target.size(0), -1, 2)
    inter_pts, inter_masks = poly_intersection(
        pred_pts, target_pts, areas1, areas2, eps)
    overlap = convex_areas(inter_pts, inter_masks)

    ious = (overlap / (areas1 + areas2 - overlap + eps)).clamp(min_v=eps)
    if linear:
        loss = 1 - ious
    else:
        loss = -ious.log()

    if weight is not None:
        loss *= weight
    
    if avg_factor is None:
        avg_factor = loss.numel()
    
    if reduction=="sum":
        return loss.sum() 
    elif reduction == "mean":
        return loss.sum()/avg_factor
    return loss


def poly_giou_loss(pred, target, eps=1e-6, weight=None, reduction='mean', avg_factor=None):
    areas1, areas2 = get_bbox_areas(pred), get_bbox_areas(target)
    pred, target = bbox2type(pred, 'poly'), bbox2type(target, 'poly')

    pred_pts = pred.view(pred.size(0), -1, 2)
    target_pts = target.view(target.size(0), -1, 2)
    inter_pts, inter_masks = poly_intersection(
        pred_pts, target_pts, areas1, areas2, eps)
    overlap = convex_areas(inter_pts, inter_masks)

    union = areas1 + areas2 - overlap + eps
    ious = (overlap / union).clamp(min=eps)

    enclose_pts, enclose_masks = poly_enclose(pred_pts, target_pts)
    enclose_areas = convex_areas(enclose_pts, enclose_masks)

    gious = ious - (enclose_areas - union) / enclose_areas
    loss = 1 - gious

    if weight is not None:
        loss *= weight
    
    if avg_factor is None:
        avg_factor = loss.numel()
    
    if reduction=="sum":
        return loss.sum() 
    elif reduction == "mean":
        return loss.sum()/avg_factor
    return loss

def poly_ciou_loss(pred, target, iou_thr, eps=1e-6, weight=None, reduction='mean', avg_factor=None):

    idx = weight > 0
    if idx.sum() == 0:
        loss = pred.mean() * 0.
        return loss

    pred = pred[idx]
    target = target[idx]

    # center diffs
    center_diffs = (pred[:, 0] - target[:, 0]).sqr() + (pred[:, 1] - target[:, 1]).sqr()

    # aspect ratio diffs
    ratio_diffs = jt.arctan(pred[:, 3] / pred[:, 2]) - jt.arctan(target[:, 3] / target[:, 2])
    ratio_diffs = 4 / np.pi / np.pi * ratio_diffs.sqr()

    # angle diffs
    angle_diffs = (pred[:, -1] - target[:, -1]) % np.pi
    angle_diffs = angle_diffs.sqr()

    #areas1, areas2 = get_bbox_areas(pred), get_bbox_areas(target)

    #pred1, target1 = bbox2type(pred, 'poly'), bbox2type(target, 'poly')
    pred = rotated_box_to_poly(pred)
    target = rotated_box_to_poly(target)

    diag = poly_enclose_diag(pred, target)
    center_diffs = center_diffs / (diag + eps)
    center_diffs = jt.clamp(center_diffs, min_v=eps, max_v=1.)

    '''
    pred_pts = pred1.view(pred1.size(0), -1, 2)
    target_pts = target1.view(target1.size(0), -1, 2)
    inter_pts, inter_masks = poly_intersection(pred_pts, target_pts, areas1, areas2, eps)
    overlap = convex_areas(inter_pts, inter_masks)

    union = areas1 + areas2 - overlap + eps
    ious2 = jt.clamp(overlap / union, min_v=eps, max_v=1.)
    '''

    ious = bbox_overlaps_rotated(pred, target).diag()

    #enclose_pts, enclose_masks = poly_enclose(pred_pts, target_pts)
    #enclose_areas = convex_areas(enclose_pts, enclose_masks)

    #gious = ious - (enclose_areas - union) / enclose_areas


    ratio_diffs = ratio_diffs / (1 - ious + ratio_diffs + 2*eps)
    ratio_diffs[ious<=iou_thr] = 0

    angle_diffs = angle_diffs / (1 - ious + angle_diffs + 2*eps)
    angle_diffs[ious<=iou_thr] = 0

    #print('ious', ious.min(), ious.max(), 'cd', center_diffs.min(), center_diffs.max(),
            #'rd', ratio_diffs.min(), ratio_diffs.max(), 'ad:', angle_diffs.min(), angle_diffs.max() )
    loss = 1 - ious + center_diffs + ratio_diffs + angle_diffs

    #if weight is not None:
        #loss *= weight
    
    if avg_factor is None:
        avg_factor = loss.numel()
    
    if reduction=="sum":
        return loss.sum() 
    elif reduction == "mean":
        return loss.sum()/max(avg_factor, 1)


    return loss


def saf(pred_oboxes, target_oboxes, n_samples=6):

    poly = rotated_box_to_poly(pred_oboxes).reshape(-1, 4, 2).roll(-1, 1)
    factors2 = jt.arange(n_samples) / n_samples
    factors1 = 1. - factors2

    center = target_oboxes[:, :2]
    cos = jt.cos(target_oboxes[:, -1]) # [n, 1]
    sin = jt.sin(target_oboxes[:, -1])
    saf = 0.

    for i1 in range(4):
        i2 = (i1 + 1) % 4
        p1 = poly[:, i1, :] # [n, 2]
        p2 = poly[:, i2, :] # [n, 2]
        pnew = p1[:, None, :] * factors1[None, :, None] + p2[:, None, :] * factors2[None, :, None] # [n, n_samples, 2]
        pnew = pnew - center[:, None, :] # [n, n_samples, 2]
        ppx =  pnew[:, :, 0] * cos[:, None] + pnew[:, :, 1] * sin[:, None] # [n, n_samples, 1]
        ppy = -pnew[:, :, 0] * sin[:, None] + pnew[:, :, 1] * cos[:, None] # [n, n_samples, 1]
        ppxy = jt.stack([ppx, ppy], dim=2) # [n, n_samples, 2]
        qqxy = jt.abs(ppxy) - 0.5 * target_oboxes[:, None, 2:4] # [n, n_samples, 2]

        sign = qqxy[:, :, 0] > 0 # [n, n_samples, 2]
        signx = jt.zeros_like(ppx)
        signx[ppx > 0] = 1.
        signx[ppx < 0] = -1.
        signy = jt.zeros_like(ppy)
        signy[ppy > 0] = 1.
        signy[ppy < 0] = -1.
        x_comp = jt.maximum(qqxy[:, :, 0], 0.) * sign * signx
        y_comp = jt.maximum(qqxy[:, :, 1], 0.) * (1. - sign) * signy

        dx = ppx[:, 1:] - ppx[:, :-1]
        dy = ppy[:, 1:] - ppy[:, :-1]

        safii = x_comp[:, :-1] * dy - y_comp[:, :-1] * dx # [n, n_samples-1]
        saf = saf + safii.sum(-1)

    return saf

def poly_saf_loss(pred, target, iou_thr, eps=1e-6, weight=None, reduction='mean', avg_factor=None):

    idx = weight > 0
    if idx.sum() == 0:
        loss = pred.mean() * 0.
        return loss

    pred = pred[idx]
    target = target[idx]

    #aa = jt.array([[-2, 3, 4, 5, 45/180*np.pi]])
    #bb = jt.array([[0,0,3,2.5,30/180*np.pi]])
    #aa1 = rotated_box_to_poly(aa).reshape(-1, 4, 2).roll(-1,1)
    #cc = saf(aa, bb)

    safs = saf(pred, target)

    pred_area = pred[:, 2] * pred[:, 3]
    target_area = target[:, 2] * target[:, 3]
    area_losses = pred_area - target_area

    # center diffs
    center_diffs = (pred[:, 0] - target[:, 0]).sqr() + (pred[:, 1] - target[:, 1]).sqr()
    center_diffs = jt.sqrt(center_diffs)

    pred_diag = jt.sqrt(pred[:, 2:4].sqr().sum(-1))
    target_diag = jt.sqrt(target[:, 2:4].sqr().sum(-1))
    dd = 0.5 * (pred_diag + target_diag)

    idx = center_diffs > dd

    center_losses = jt.abs(pred[:, :2] - target[:, :2]).sum(-1)
    center_losses = jt.clamp(center_losses, max_v=4.)

    num = 2 * target_area + area_losses
    den = target_area + (1-idx) * safs + idx * pred_area
    #loss = (2 - num / den) * (1-idx) + center_losses * idx
    #loss = (2 - num / den) * (1-idx)
    iou = num / den - 1
    loss = 1 - iou + center_losses * center_losses / (1 - iou + center_losses)

    #if weight is not None:
        #loss *= weight
    
    if avg_factor is None:
        avg_factor = loss.numel()
    
    if reduction=="sum":
        return loss.sum() 
    elif reduction == "mean":
        return loss.sum()/max(avg_factor, 1)

    return loss

def poly_sdf_loss(pred,target,iou_thr,weight=None,eps=1e-6,avg_factor=None,reduction="mean"):

    idx = weight > 0
    if idx.sum() == 0:
        loss = pred.mean() * 0.
        return loss

    pred = pred[idx, :]
    target = target[idx, :]

    pred_poly = rotated_box_to_poly(pred).reshape(-1, 4, 2)
    n_sample = 40
    factors = jt.arange(n_sample) / n_sample
    n_factors = 1. - factors
    pred_poly2 = []
    for i1 in range(4):
        i2 = (i1 + 1) % 4
        p1 = pred_poly[:, i1, :]
        p2 = pred_poly[:, i2, :]
        pnew = p1[:, None, :] * factors[None, :, None] + p2[:, None, :] * n_factors[None, :, None]
        pred_poly2.append(pnew)

    pred_poly2 = jt.concat(pred_poly2, dim=1)

    center = target[:, :2]
    qq = pred_poly2 - center[:, None, :]
    cos = jt.cos(target[:,-1])
    sin = jt.sin(target[:,-1])
    qqx = qq[:, :, 0] * cos[:, None] + qq[:, :, 1] * sin[:, None]
    qqy = -qq[:, :, 0] * sin[:, None] + qq[:, :, 1] * cos[:, None]
    qqxy = jt.stack([qqx, qqy], dim=2)
    qqxy = jt.abs(qqxy) - 0.5 * target[:, None, 2:4]
    qqxy1 = jt.maximum(qqxy, 0.)
    loss = jt.sqrt(qqxy1[:, :, 0].sqr() + qqxy1[:, :, 1].sqr())
    #loss = loss + jt.minimum(jt.abs(jt.max(qqxy, dim=-1)), 0.0)
    loss = loss + jt.abs(jt.minimum(jt.max(qqxy, dim=-1), 0.0))

    loss = loss.mean(dim=1)

    target_poly = rotated_box_to_poly(target)
    diag = poly_enclose_diag(pred_poly, target_poly)

    loss = loss / diag
    loss = jt.clamp(loss, max_v=3.)
    #scale = target[:, 2] + target[:, 3]

    area_loss = jt.abs(pred[:, 2] * pred[:,3] - target[:, 2] * target[:, 3])
    area_loss = area_loss / diag
    area_loss = jt.clamp(area_loss, max_v=3.)
    center_loss = jt.abs(pred[:, :2] - target[:, :2]).sum(-1)
    center_loss = center_loss / diag
    center_loss = jt.clamp(center_loss, max_v=3.)
    loss = loss + 0.1*area_loss + 1*center_loss

    if avg_factor is None:
        avg_factor = max(loss.shape[0],1)

    if reduction == "mean":
        loss = loss.sum()/max(avg_factor,1)
    elif reduction == "sum":
        loss = loss.sum()

    #if loss > 1000:
        #__import__('pdb').set_trace()

    return loss

@LOSSES.register_module()
class PolyIoULoss(nn.Module):

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(PolyIoULoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if weight is not None and weight.ndim > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * poly_iou_loss(
            pred,
            target,
            weight=weight,
            linear=self.linear,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class PolyGIoULoss(nn.Module):

    def __init__(self,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(PolyGIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not jt.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * poly_giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@LOSSES.register_module()
class PolyCIoULoss(nn.Module):

    def __init__(self,
                 eps=1e-6,
                 reduction='mean',
                 iou_thr=0.2,
                 loss_weight=1.0):
        super(PolyCIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.iou_thr = iou_thr
        self.loss_weight = loss_weight

    def execute(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        #if (weight is not None) and (not jt.any(weight > 0)) and (
                #reduction != 'none'):
            #return (pred * weight).sum()  # 0
        if weight is not None and weight.ndim > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss1 = self.loss_weight * poly_saf_loss(
            pred,
            target,
            self.iou_thr,
            weight=weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss1
