    def forward_single_val(self, x, stride):

        fam_reg_feat = x
        for fam_reg_conv in self.fam_reg_convs:
            fam_reg_feat = fam_reg_conv(fam_reg_feat) # [B, 256, 128, 128]
        fam_bbox_pred = self.fam_reg(fam_reg_feat) # [B, 5*n_anchors, 128, 128]

        B = x.shape[0]
        featmap_size = tuple(fam_bbox_pred.shape[-2:])
        #featmap_n = featmap_size[0] * featmap_size[1]
        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).reshape(B, -1, self.n_anchors, 5)
        # [B, 128*128,n_anchors,5]

        # only forward during training
        #if self.is_training():
            #fam_cls_feat = x
            #for fam_cls_conv in self.fam_cls_convs:
                #fam_cls_feat = fam_cls_conv(fam_cls_feat) # [B, 256, 128, 128]
            #fam_cls_score = self.fam_cls(fam_cls_feat) # [B, 10*n_anchors, 128, 128]
        #else:
            #fam_cls_score = None

        #label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        #fam_cls_score = fam_cls_score.permute(0, 2, 3, 1).reshape(B, -1, self.n_anchors, label_channels)

        num_level = self.anchor_strides.index(stride)
        anchor_list = self.anchor_generators[num_level].grid_anchors(
                featmap_size, self.anchor_strides[num_level])

        # refine anchors using fam
        refine_anchor = []
        all_odm_cls_feat = []
        all_odm_reg_feat = []

        for ii in range(self.n_anchors):

            famii = fam_bbox_pred[:, :, ii, :]  # [B, 128*128, 5]

            anchorii = anchor_list.clone()  # [128*128*n_anchors, 5]
            anchorii = anchorii.reshape(-1, self.n_anchors, 5)
            anchorii = anchorii[:, ii, :]

            refine_anchorii = bbox_decode(
                    featmap_size,
                    famii.detach(),
                    anchorii,
                    self.target_means,
                    self.target_stds,
                    neg_indices = None) # [B, 128, 128, 5]

            align_feat = self.align_conv(x, refine_anchorii.clone(), stride) # [B, 256, 128, 128]

            or_feat = self.or_conv(align_feat) # [2, 256, 128, 128]
            odm_reg_feat = or_feat
            if self.with_orconv:
                odm_cls_feat = self.or_pool(or_feat)
            else:
                odm_cls_feat = or_feat

            refine_anchor.append(refine_anchorii)
            all_odm_cls_feat.append(odm_cls_feat)
            all_odm_reg_feat.append(odm_reg_feat)

        refine_anchor = jt.stack(refine_anchor, 3) # [B, 128, 128, n_anchors, 5]
        refine_anchor = refine_anchor.reshape(B, -1, 5)
        odm_cls_feat = jt.concat(all_odm_cls_feat, 1) # [B, n_anchors*32, 128, 128]
        odm_reg_feat = jt.concat(all_odm_reg_feat, 1) # [B, n_anchors*256, 128, 128]


        for odm_reg_conv in self.odm_reg_convs:
            odm_reg_feat = odm_reg_conv(odm_reg_feat)
        for odm_cls_conv in self.odm_cls_convs:
            odm_cls_feat = odm_cls_conv(odm_cls_feat)
        odm_cls_score = self.odm_cls(odm_cls_feat)  # [B, n_anchors*10, 128, 128]
        odm_bbox_pred = self.odm_reg(odm_reg_feat)  # [B, n_anchors*5, 128, 128]

        odm_cls_score = self.odm_cls(odm_cls_feat)  # [B, n_anchors*10, 128, 128]
        odm_bbox_pred = self.odm_reg(odm_reg_feat)  # [B, n_anchors*5, 128, 128]


        return refine_anchor, odm_cls_score, odm_bbox_pred
