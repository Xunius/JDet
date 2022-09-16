import numpy as np
import jittor as jt
from jittor import nn

from jdet.models.utils.weight_init import normal_init,bias_init_with_prob
from jdet.models.utils.modules import ConvModule
from jdet.utils.general import multi_apply
from jdet.utils.registry import HEADS,LOSSES,BOXES,build_from_cfg


# from jdet.ops.dcn_v2 import DeformConv
from jdet.ops.dcn_v1 import DeformConv
from jdet.ops.orn import ORConv2d, RotationInvariantPooling
from jdet.ops.nms_rotated import multiclass_nms_rotated, singleclass_nms_rotated
from jdet.models.boxes.box_ops import delta2bbox_rotated, rotated_box_to_poly
from jdet.models.boxes.anchor_target import images_to_levels,anchor_target,anchor_target_single,initial_assign
from jdet.models.boxes.anchor_generator import AnchorGeneratorRotatedS2ANet


@HEADS.register_module()
class S2ANetHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 with_orconv=True,
                 anchor_scales=[4],
                 anchor_ratios=[1.0],
                 anchor_strides=[8, 16, 32, 64, 128],
                 anchor_angles=[0,30,60,90,120,150],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                 loss_fam_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_fam_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_odm_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_odm_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 test_cfg=dict(
                    nms_pre=2000,
                    min_bbox_size=0,
                    score_thr=0.05,
                    nms=dict(type='nms_rotated', iou_thr=0.1),
                    max_per_img=2000),
                train_cfg=dict(
                    fam_cfg=dict(
                        assigner=dict(
                            type='MaxIoUAssigner',
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.4,
                            min_pos_iou=0,
                            ignore_iof_thr=-1,
                            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                        bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                                        target_means=(0., 0., 0., 0., 0.),
                                        target_stds=(1., 1., 1., 1., 1.),
                                        clip_border=True),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False),
                    odm_cfg=dict(
                        assigner=dict(
                            type='MaxIoUAssigner',
                            pos_iou_thr=0.5,
                            neg_iou_thr=0.4,
                            min_pos_iou=0,
                            ignore_iof_thr=-1,
                            iou_calculator=dict(type='BboxOverlaps2D_rotated')),
                        bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                                        target_means=(0., 0., 0., 0., 0.),
                                        target_stds=(1., 1., 1., 1., 1.),
                                        clip_border=True),
                        allowed_border=-1,
                        pos_weight=-1,
                        debug=False))):
        super(S2ANetHead, self).__init__()
        self.num_classes = num_classes  # 11
        self.in_channels = in_channels  # 256
        self.feat_channels = feat_channels # 256
        self.stacked_convs = stacked_convs # 2
        self.with_orconv = with_orconv     # True
        self.anchor_scales = anchor_scales   # [4]
        self.anchor_ratios = anchor_ratios   # [1.0]
        self.anchor_strides = anchor_strides  # [8, 16, 32, 64, 128]
        self.anchor_angles = np.array(anchor_angles) / 180 * np.pi
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means  # [0.0, ]*5
        self.target_stds = target_stds    # [1.0, ]*5

        self.use_sigmoid_cls = loss_odm_cls.get('use_sigmoid', False)
        self.sampling = loss_odm_cls['type'] not in ['FocalLoss', 'GHMC'] # False
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))
        self.loss_fam_cls = build_from_cfg(loss_fam_cls,LOSSES)
        self.loss_fam_bbox = build_from_cfg(loss_fam_bbox,LOSSES)
        self.loss_odm_cls = build_from_cfg(loss_odm_cls,LOSSES)
        self.loss_odm_bbox = build_from_cfg(loss_odm_bbox,LOSSES)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(AnchorGeneratorRotatedS2ANet(anchor_base, anchor_scales, anchor_ratios, self.anchor_angles))

        # anchor cache
        self.base_anchors = dict()
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU()
        self.fam_reg_convs = nn.ModuleList()
        self.fam_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.fam_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    norm_cfg={'type': 'BN2d'},
                    padding=1))
            self.fam_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    norm_cfg={'type': 'BN2d'},
                    padding=1))

        #self.fam_reg = nn.Conv2d(self.feat_channels, 5, 1) # may need to changes this
        self.n_anchors = len(self.anchor_scales)*len(self.anchor_ratios)*len(self.anchor_angles)
        self.fam_reg = nn.Conv2d(self.feat_channels, 5*self.n_anchors, 1)
        self.fam_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)

        self.align_conv = AlignConv(
            self.feat_channels, self.feat_channels, kernel_size=3)

        if self.with_orconv:
            self.or_conv = ORConv2d(self.feat_channels, int(
                self.feat_channels / 8), kernel_size=3, padding=1, arf_config=(1, 8))
        else:
            self.or_conv = nn.Conv2d(
                self.feat_channels, self.feat_channels, 3, padding=1)
        #self.or_pool = RotationInvariantPooling(256, 8)
        self.or_pool = RotationInvariantPooling(self.feat_channels, 8)  # replace 256 with self.feat_channels

        self.odm_reg_convs = nn.ModuleList()
        self.odm_cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = int(self.feat_channels /
                      8) if i == 0 and self.with_orconv else self.feat_channels
            self.odm_reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    norm_cfg={'type': 'BN2d'},
                    padding=1))
            self.odm_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    norm_cfg={'type': 'BN2d'},
                    padding=1))

        self.odm_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.odm_reg = nn.Conv2d(self.feat_channels, 5, 3, padding=1)

        self.init_weights()

    def init_weights(self):
        for m in self.fam_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.fam_cls_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fam_reg, std=0.01)
        normal_init(self.fam_cls, std=0.01, bias=bias_cls)

        self.align_conv.init_weights()

        normal_init(self.or_conv, std=0.01)
        for m in self.odm_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.odm_cls_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.odm_cls, std=0.01, bias=bias_cls)
        normal_init(self.odm_reg, std=0.01)



    def forward_single_train(self, x, stride, targets=None):
        fam_reg_feat = x
        for fam_reg_conv in self.fam_reg_convs:
            fam_reg_feat = fam_reg_conv(fam_reg_feat) # [B, 256, 128, 128]
        fam_bbox_pred = self.fam_reg(fam_reg_feat) # [B, 5*n_anchors, 128, 128]

        B = x.shape[0]
        featmap_size = tuple(fam_bbox_pred.shape[-2:])
        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, self.n_anchors, 5)
        # [B, n_anchors, 5, 128, 128]

        # only forward during training
        if self.is_training():
            fam_cls_feat = x
            for fam_cls_conv in self.fam_cls_convs:
                fam_cls_feat = fam_cls_conv(fam_cls_feat) # [B, 256, 128, 128]
            fam_cls_score = self.fam_cls(fam_cls_feat) # [B, 10, 128, 128]
        else:
            fam_cls_score = None

        gt_bboxes,gt_labels,img_metas,gt_bboxes_ignore = targets
        num_level = self.anchor_strides.index(stride)
        __import__('pdb').set_trace()
        anchors, valid_flags = self.get_init_anchors2(num_level, featmap_size, img_metas)

        cfg = self.train_cfg.copy()

        # select anchors by ratio and angle
        # loop through imgs
        aidx_arr = multi_apply(
                initial_assign, anchors, valid_flags,
                gt_bboxes, gt_bboxes_ignore, img_metas,
                n_ratios=len(self.anchor_ratios),
                n_angles=len(self.anchor_angles),
                n_scales=len(self.anchor_scales),
                cfg=cfg.fam_cfg)
        aidx_arr = np.array(aidx_arr).T

        # loop through imgs
        anchor_idx = []
        refine_anchor = []
        for ii in range(B):

            # select the closest anchor
            sel_anchorii = anchors[ii].reshape(-1, self.n_anchors, 5)
            sel_idxii = aidx_arr[ii]
            sel_anchorii = sel_anchorii[jt.arange(0, len(sel_idxii)), sel_idxii]
            famii = fam_bbox_pred[ii, jt.arange(0, len(sel_idxii)), sel_idxii]

            # add anchor priors to model predicted offsets, get bboxes in absolute coordinate
            refine_anchorii = bbox_decode(
                    famii.detach().permute(1, 0).reshape(1, 5, *featmap_size),
                    sel_anchorii,
                    self.target_means,
                    self.target_stds)

            refine_anchor.append(refine_anchorii)
            anchor_idx.append(sel_idxii)
            fam_bbox_pred[ii,:,0] = famii

        # pass refined anchors to aligned conv module
        refine_anchor = jt.concat(refine_anchor, dim=0)
        align_feat = self.align_conv(x, refine_anchor.clone(), stride)
        # select only 1 anchor prediction from the FAM predictions
        fam_bbox_pred = fam_bbox_pred[:,:,0,:].reshape(B, *featmap_size, 5).permute(0, 3, 1, 2)
        # [B, 5, H, W]

        or_feat = self.or_conv(align_feat)
        odm_reg_feat = or_feat
        if self.with_orconv:
            odm_cls_feat = self.or_pool(or_feat)
        else:
            odm_cls_feat = or_feat

        for odm_reg_conv in self.odm_reg_convs:
            odm_reg_feat = odm_reg_conv(odm_reg_feat)
        for odm_cls_conv in self.odm_cls_convs:
            odm_cls_feat = odm_cls_conv(odm_cls_feat)
        odm_cls_score = self.odm_cls(odm_cls_feat)
        odm_bbox_pred = self.odm_reg(odm_reg_feat)

        return fam_cls_score, fam_bbox_pred, refine_anchor, odm_cls_score, odm_bbox_pred, anchor_idx

    def forward_multi_val(self, x, img_metas):
        '''process a batch in evaluation mode

        x: tuple, predictions at different strides
        '''

        #bboxes = []
        fam_cls_scores = []
        fam_bbox_preds = []
        refine_anchors = []
        odm_cls_scores = []
        odm_bbox_preds = []
        # loop through anchor priors
        for jj in range(self.n_anchors):
            outsjj = multi_apply(self.forward_single_val, x, self.anchor_strides, aidx=jj)
            fam_cls_scores.append(outsjj[0])
            fam_bbox_preds.append(outsjj[1])
            refine_anchors.append(outsjj[2])
            odm_cls_scores.append(outsjj[3])
            odm_bbox_preds.append(outsjj[4])

        bboxes = self.get_bboxes2(fam_cls_scores, fam_bbox_preds, refine_anchors, odm_cls_scores, odm_bbox_preds, img_metas)
        #print('len(bboxes)=', len(bboxes[0][0]))
        #bboxes.extend(bboxesjj)

        return bboxes


    def forward_single_val(self, x, stride, aidx):

        fam_reg_feat = x
        for fam_reg_conv in self.fam_reg_convs:
            fam_reg_feat = fam_reg_conv(fam_reg_feat) # [B, 256, 128, 128]
        fam_bbox_pred = self.fam_reg(fam_reg_feat) # [B, 5*n_anchors, 128, 128]

        B = x.shape[0]
        featmap_size = tuple(fam_bbox_pred.shape[-2:])
        fam_bbox_pred = fam_bbox_pred.reshape(B, -1, 5, *featmap_size)
        # [B, n_anchors, 5, 128, 128]
        fam_bbox_pred = fam_bbox_pred[:, aidx]
        # [B, 5, 128, 128]

        # only forward during training
        if self.is_training():
            fam_cls_feat = x
            for fam_cls_conv in self.fam_cls_convs:
                fam_cls_feat = fam_cls_conv(fam_cls_feat) # [B, 256, 128, 128]
            fam_cls_score = self.fam_cls(fam_cls_feat) # [B, 10, 128, 128]
        else:
            fam_cls_score = None

        #featmap_size = tuple(fam_bbox_pred.shape[-2:])
        num_level = self.anchor_strides.index(stride)
        init_anchor = self.anchor_generators[num_level].grid_anchors(
                featmap_size, self.anchor_strides[num_level])

        sel_anchor = init_anchor[aidx::self.n_anchors]
        refine_anchor = bbox_decode(
                fam_bbox_pred.detach(),
                sel_anchor,
                self.target_means,
                self.target_stds)

        align_feat = self.align_conv(x, refine_anchor.clone(), stride)

        or_feat = self.or_conv(align_feat)
        odm_reg_feat = or_feat
        if self.with_orconv:
            odm_cls_feat = self.or_pool(or_feat)
        else:
            odm_cls_feat = or_feat

        for odm_reg_conv in self.odm_reg_convs:
            odm_reg_feat = odm_reg_conv(odm_reg_feat)
        for odm_cls_conv in self.odm_cls_convs:
            odm_cls_feat = odm_cls_conv(odm_cls_feat)
        odm_cls_score = self.odm_cls(odm_cls_feat)
        odm_bbox_pred = self.odm_reg(odm_reg_feat)

        return fam_cls_score, fam_bbox_pred, refine_anchor, odm_cls_score, odm_bbox_pred



    def get_init_anchors(self,
                         featmap_sizes,
                         img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        anchor_list = []
        for j in range(num_imgs):
            multi_level_anchors = []
            for i in range(num_levels):
                anchors = self.anchor_generators[i].grid_anchors(featmap_sizes[i], self.anchor_strides[i])
                multi_level_anchors.append(anchors)
            anchor_list.append(multi_level_anchors)
        #anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                w,h = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list


    def get_init_anchors2(self,
                         stride_idx,
                         featmap_size,
                         img_metas):
        """Get anchors according to feature map sizes.

        Args:
            stride_idx (int): index of stride level.
            featmap_size (tuple): feature map size of a level.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        #num_levels = len(featmap_sizes)
        #num_levels = 1

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        anchor_list = []
        for j in range(num_imgs):
            #multi_level_anchors = []
            #for i in range(num_levels):
            anchors = self.anchor_generators[stride_idx].grid_anchors(
                    featmap_size, self.anchor_strides[stride_idx])
            #multi_level_anchors.append(anchors)
            anchor_list.append(anchors)
        #anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            #multi_level_flags = []
            #for i in range(num_levels):
            anchor_stride = self.anchor_strides[stride_idx]
            feat_h, feat_w = featmap_size
            w,h = img_meta['pad_shape'][:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
            flags = self.anchor_generators[stride_idx].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
            #multi_level_flags.append(flags)
            valid_flag_list.append(flags)
        return anchor_list, valid_flag_list

    def get_refine_anchors(self,
                           featmap_sizes,
                           refine_anchors,
                           img_metas,
                           is_train=True):
        num_levels = len(featmap_sizes)

        refine_anchors_list = []
        for img_id, img_meta in enumerate(img_metas):
            mlvl_refine_anchors = []
            for i in range(num_levels):
                refine_anchor = refine_anchors[i][img_id].reshape(-1, 5)
                #refine_anchor = refine_anchors[i][:,img_id].reshape(-1, 5)
                mlvl_refine_anchors.append(refine_anchor)
            refine_anchors_list.append(mlvl_refine_anchors)

        valid_flag_list = []
        if is_train:
            for img_id, img_meta in enumerate(img_metas):
                multi_level_flags = []
                for i in range(num_levels):
                    anchor_stride = self.anchor_strides[i]
                    feat_h, feat_w = featmap_sizes[i]
                    w,h = img_meta['pad_shape'][:2]
                    valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                    valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                    flags = self.anchor_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                    flags = flags[::self.n_anchors]
                    multi_level_flags.append(flags)
                valid_flag_list.append(multi_level_flags)
        return refine_anchors_list, valid_flag_list

    def loss(self,
             fam_cls_scores,
             fam_bbox_preds,
             refine_anchors,
             odm_cls_scores,
             odm_bbox_preds,
             anchor_idx,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        cfg = self.train_cfg.copy()
        featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_init_anchors(featmap_sizes, img_metas)

        # anchor number of multi levels
        #num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]] # [16384, 4096, 1024, 256, 64]
        num_level_anchors = [anchors.size(0)//self.n_anchors for anchors in anchor_list[0]] # [16384, 4096, 1024, 256, 64]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)): # loop through images
            tmp_list = []
            for j in range(len(anchor_list[i])): # loop through strides
                anchor_idxij = anchor_idx[j][i]
                tmp = anchor_list[i][j]
                tmp = tmp.reshape(-1, self.n_anchors, 5)
                tmp = tmp[jt.arange(0, tmp.shape[0]), anchor_idxij]

                flag = valid_flag_list[i][j]
                flag = flag.reshape(-1, self.n_anchors)
                flag = flag[jt.arange(0, tmp.shape[0]), anchor_idxij]

                tmp_list.append(tmp)
                anchor_list[i][j] = tmp
                valid_flag_list[i][j] = flag

            #concat_anchor_list.append(jt.contrib.concat(anchor_list[i])) # sum of [16384, 4096, 1024, 256, 64]
            concat_anchor_list.append(jt.contrib.concat(tmp_list)) # sum of [16384, 4096, 1024, 256, 64]
        all_anchor_list = images_to_levels(concat_anchor_list,num_level_anchors) # 5-list, each [B, Nanchors, 5]

        # Feature Alignment Module
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg.fam_cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,num_total_pos, num_total_neg = cls_reg_targets

        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos

        losses_fam_cls, losses_fam_bbox = multi_apply(
            self.loss_fam_single,
            fam_cls_scores,
            fam_bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg.fam_cfg)

        # Oriented Detection Module targets
        refine_anchors_list, valid_flag_list = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0)
                             for anchors in refine_anchors_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(refine_anchors_list)):
            concat_anchor_list.append(jt.contrib.concat(refine_anchors_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            refine_anchors_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg.odm_cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_odm_cls, losses_odm_bbox = multi_apply(
            self.loss_odm_single,
            odm_cls_scores,
            odm_bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg.odm_cfg)

        return dict(loss_fam_cls=losses_fam_cls,
                    loss_fam_bbox=losses_fam_bbox,
                    loss_odm_cls=losses_odm_cls,
                    loss_odm_bbox=losses_odm_bbox)

    def loss_fam_single(self,
                        fam_cls_score,
                        fam_bbox_pred,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        num_total_samples,
                        cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        fam_cls_score = fam_cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_fam_cls = self.loss_fam_cls(
            fam_cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)

        ## weight up angle loss
        #bbox_weights[:, -1] = 8.0 * bbox_weights[:, -1]

        if reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_coder_cfg = cfg.get('bbox_coder', '')
            if bbox_coder_cfg == '':
                bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
            bbox_coder = build_from_cfg(bbox_coder_cfg,BOXES)
            anchors = anchors.reshape(-1, 5)
            fam_bbox_pred = bbox_coder.decode(anchors, fam_bbox_pred)
        loss_fam_bbox = self.loss_fam_bbox(
            fam_bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_fam_cls, loss_fam_bbox

    def loss_odm_single(self,
                        odm_cls_score,
                        odm_bbox_pred,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        num_total_samples,
                        cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        odm_cls_score = odm_cls_score.permute(0, 2, 3,
                                              1).reshape(-1, self.cls_out_channels)
        loss_odm_cls = self.loss_odm_cls(
            odm_cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        odm_bbox_pred = odm_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)

        ## weight up angle loss
        #bbox_weights[:, -1] = 8.0 * bbox_weights[:, -1]

        if reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_coder_cfg = cfg.get('bbox_coder', '')
            if bbox_coder_cfg == '':
                bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
            bbox_coder = build_from_cfg(bbox_coder_cfg,BOXES)
            anchors = anchors.reshape(-1, 5)
            odm_bbox_pred = bbox_coder.decode(anchors, odm_bbox_pred)
        loss_odm_bbox = self.loss_odm_bbox(
            odm_bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_odm_cls, loss_odm_bbox

    def get_bboxes(self,
                   fam_cls_scores,
                   fam_bbox_preds,
                   refine_anchors,
                   odm_cls_scores,
                   odm_bbox_preds,
                   img_metas,
                   rescale=True):
        assert len(odm_cls_scores) == len(odm_bbox_preds)
        cfg = self.test_cfg.copy()

        featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_scores]
        num_levels = len(odm_cls_scores)

        refine_anchors = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas, is_train=False)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                odm_cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                odm_bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               refine_anchors[0][img_id], img_shape,
                                               scale_factor, cfg, rescale)

            result_list.append(proposals)
        return result_list


    def get_bboxes2(self,
            fam_cls_scores,
            fam_bbox_preds,
            refine_anchors,
            odm_cls_scores,
            odm_bbox_preds,
            img_metas, rescale=True):

        #assert len(odm_cls_scores) == len(odm_bbox_preds)
        cfg = self.test_cfg.copy()

        featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_scores[0]]
        num_levels = len(odm_cls_scores[0])

        #refine_anchors = self.get_refine_anchors(
            #featmap_sizes, refine_anchors, img_metas, is_train=False)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = []
            bbox_pred_list = []
            refine_anchor_list = []
            for sidx in range(num_levels):
                xlist = []
                ylist = []
                alist = []
                for aidx in range(self.n_anchors):
                    xii = odm_cls_scores[aidx][sidx][img_id].detach()
                    yii = odm_bbox_preds[aidx][sidx][img_id].detach()
                    ranchors = refine_anchors[aidx]
                    aii = self.get_refine_anchors([featmap_sizes[sidx]], [ranchors[sidx][img_id:img_id+1]], [img_metas[img_id]], is_train=False)
                    xlist.append(xii)
                    ylist.append(yii)
                    alist.append(aii[0][0][0])

                cls_score_list.append(xlist)
                bbox_pred_list.append(ylist)
                refine_anchor_list.append(alist)

                #odm_cls_scores[i][img_id].detach() for i in range(num_levels)
            #]
            #bbox_pred_list = [
                #odm_bbox_preds[i][img_id].detach() for i in range(num_levels)
            #]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single2(cls_score_list, bbox_pred_list,
                                               refine_anchor_list, img_shape,
                                               scale_factor, cfg, rescale)

            result_list.append(proposals)
        return result_list

    def get_bboxes_single2(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """

        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        # loop through strides
        for cls_score_s, bbox_pred_s, anchors_s in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            # loop through anchor priors
            #bboxes_tmp = []
            #scores_tmp = []
            cls_score = jt.stack(cls_score_s, 0)
            bbox_pred = jt.stack(bbox_pred_s, 0)
            anchors = jt.stack(anchors_s, 0)
            #for cls_score, bbox_pred, anchors in zip(cls_score_s, bbox_pred_s, anchors_s):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute( 0, 2, 3, 1).reshape(-1, self.cls_out_channels) # [10,128,128]->[128*128,10]
                    #1, 2, 0).reshape(-1, self.cls_out_channels) # [10,128,128]->[128*128,10]

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            #bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5) #[5,128,128]->[128*128,5]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5) #[5,128,128]->[128*128,5]
            anchors = anchors.reshape(-1, 5)
            # anchors = rect2rbox(anchors)
            #nms_pre = cfg.get('nms_pre', -1) // self.n_anchors
            nms_pre = cfg.get('nms_pre', -1)
            #print('nms_pre:', nms_pre)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores = scores.max(dim=1)
                else:
                    max_scores = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                #topk_inds = max_scores >= cfg.get('score_thr', 0.1)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox_rotated(anchors, bbox_pred, self.target_means,
                                        self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            #bboxes_tmp.append(bboxes)
            #scores_tmp.append(scores)

            #mlvl_bboxes.extend(bboxes_tmp)
            #mlvl_scores.extend(scores_tmp)

        mlvl_bboxes = jt.contrib.concat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes[..., :4] /= scale_factor
        mlvl_scores = jt.contrib.concat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = jt.zeros((mlvl_scores.shape[0], 1),dtype=mlvl_scores.dtype)
            mlvl_scores = jt.contrib.concat([padding, mlvl_scores], dim=1)

        #det_bboxes, det_labels = multiclass_nms_rotated(mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        #score_factors = jt.array([0.9, 1.1, 1.1, 1, 0.9, 1, 0.9, 1.1, 0.9, 1.1])
        score_factors = None
        det_bboxes, det_labels = singleclass_nms_rotated(mlvl_bboxes, mlvl_scores,
                cfg.score_thr, cfg.nms, cfg.max_per_img, score_factors=score_factors)

        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        polys = rotated_box_to_poly(boxes)
        return polys, scores, det_labels

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """

        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            # anchors = rect2rbox(anchors)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores = scores.max(dim=1)
                else:
                    max_scores = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox_rotated(anchors, bbox_pred, self.target_means,
                                        self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = jt.contrib.concat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes[..., :4] /= scale_factor
        mlvl_scores = jt.contrib.concat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = jt.zeros((mlvl_scores.shape[0], 1),dtype=mlvl_scores.dtype)
            mlvl_scores = jt.contrib.concat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(mlvl_bboxes,
                                                        mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)

        boxes = det_bboxes[:, :5]
        scores = det_bboxes[:, 5]
        polys = rotated_box_to_poly(boxes)
        return polys, scores, det_labels


    def parse_targets(self,targets,is_train=True):
        img_metas = []
        gt_bboxes = []
        gt_bboxes_ignore = []
        gt_labels = []

        for target in targets:
            if is_train:
                gt_bboxes.append(target["rboxes"])
                gt_labels.append(target["labels"])
                gt_bboxes_ignore.append(target["rboxes_ignore"])
            img_metas.append(dict(
                img_shape=target["img_size"][::-1],
                scale_factor=target["scale_factor"],
                pad_shape = target["pad_shape"]
            ))
        if not is_train:
            return img_metas
        return gt_bboxes,gt_labels,img_metas,gt_bboxes_ignore

    def execute(self, feats,targets):
        #outs = multi_apply(self.forward_single, feats, self.anchor_strides)
        if self.is_training():
            parsed_targets = self.parse_targets(targets)
            outs = multi_apply(self.forward_single_train, feats, self.anchor_strides,
                    targets=parsed_targets)
            #return self.loss(*outs,*self.parse_targets(targets))
            return self.loss(*outs,*parsed_targets)
        else:
            bboxes = self.forward_multi_val(feats, img_metas=self.parse_targets(targets, is_train=False))
            #outs = multi_apply(self.forward_single_val, feats, self.anchor_strides,
                    #img_metas=self.parse_targets(targets,is_train=False))
            #return self.get_bboxes(*outs,self.parse_targets(targets,is_train=False))
            return bboxes

def bbox_decode(
        bbox_preds,
        anchors,
        means=[0, 0, 0, 0, 0],
        stds=[1, 1, 1, 1, 1]):
    """
    Decode bboxes from deltas
    :param bbox_preds: [N,5,H,W]
    :param anchors: [H*W,5]
    :param means: mean value to decode bbox
    :param stds: std value to decode bbox
    :return: [N,H,W,5]
    """
    num_imgs, _, H, W = bbox_preds.shape
    bboxes_list = []
    for img_id in range(num_imgs):
        bbox_pred = bbox_preds[img_id]
        # bbox_pred.shape=[5,H,W]
        bbox_delta = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
        bboxes = delta2bbox_rotated(anchors, bbox_delta, means, stds, wh_ratio_clip=1e-6)
        bboxes = bboxes.reshape(H, W, 5)
        bboxes_list.append(bboxes)
    return jt.stack(bboxes_list, dim=0)


class AlignConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.deform_conv = DeformConv(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2,
                                      deformable_groups=deformable_groups)
        self.relu = nn.ReLU()

    def init_weights(self):
        normal_init(self.deform_conv, std=0.01)

    @jt.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        dtype = anchors.dtype
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = jt.arange(-pad, pad + 1, dtype=dtype)
        yy, xx = jt.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = jt.arange(0, feat_w, dtype=dtype)
        yc = jt.arange(0, feat_h, dtype=dtype)
        yc, xc = jt.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx # [feat_h*feat_w, 9], [128*128, 9]
        y_conv = yc[:, None] + yy

        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = jt.unbind(anchors, dim=1) # anchors: [feat_h*feat_w, 5]
        x_ctr, y_ctr, w, h = x_ctr / stride, y_ctr / stride, w / stride, h / stride
        #print('angle: ', a)
        cos, sin = jt.cos(a), jt.sin(a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = jt.stack([offset_y, offset_x], dim=-1)
        # NA,ks*ks*2
        offset = offset.reshape(anchors.size(
            0), -1).permute(1, 0).reshape(-1, feat_h, feat_w) # [9*2, 128, 128]
        return offset





    def execute(self, x, anchors, stride):
        num_imgs, H, W = anchors.shape[:3]
        offset_list = [
            self.get_offset(anchors[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor = jt.stack(offset_list, dim=0) # [B, 18, 128, 128]
        x = self.relu(self.deform_conv(x, offset_tensor)) # same as input x
        return x

def draw_box(box):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    x0, y0, w, h, a = np.array(box)
    rot1 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rot2 = rot1.T
    p1 = np.array([x0 - w/2, y0 - h/2])
    p2 = np.array([x0 + w/2, y0 - h/2])
    p3 = np.array([x0 + w/2, y0 + h/2])
    p4 = np.array([x0 - w/2, y0 + h/2])

    p1r1 = np.dot(p1[None, :], rot1)[0]
    p2r1 = np.dot(p2[None, :], rot1)[0]
    p3r1 = np.dot(p3[None, :], rot1)[0]
    p4r1 = np.dot(p4[None, :], rot1)[0]

    p1r2 = np.dot(p1[None, :], rot2)[0]
    p2r2 = np.dot(p2[None, :], rot2)[0]
    p3r2 = np.dot(p3[None, :], rot2)[0]
    p4r2 = np.dot(p4[None, :], rot2)[0]

    box1 = [p1r1, p2r1, p3r1, p4r1]
    box2 = [p1r2, p2r2, p3r2, p4r2]

    for ii in range(4):
        ax.plot(box1[ii][0], box1[ii][1], 'bo', label='1')
        ax.plot(box2[ii][0], box2[ii][1], 'ro', label='2')
        ax.text(box1[ii][0], box1[ii][1], str(ii), color='b')
        ax.text(box2[ii][0], box2[ii][1], str(ii), color='r')

    fig.show()
    return fig, ax

def draw_anchor(xx, yy):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    xx = np.array(xx)
    yy = np.array(yy)
    ax.scatter(xx, yy, s=10)
    fig.show()
    return fig, ax
