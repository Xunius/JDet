import jittor as jt 
from jittor import nn 

from jdet.utils.registry import MODELS,build_from_cfg,BACKBONES,HEADS,NECKS


@MODELS.register_module()
class S2ANet(nn.Module):
    """
    """

    def __init__(self,backbone,neck=None,roi_head=None):
        super(S2ANet,self).__init__()
        self.backbone = build_from_cfg(backbone,BACKBONES)
        self.neck = build_from_cfg(neck,NECKS)
        self.roi_heads = build_from_cfg(roi_head,HEADS)

    def execute(self,images,targets):
        '''
        Args:
            images (jt.Var): image tensors, shape is [N,C,H,W]
            targets (list[dict]): targets for each image
        Rets:
            outputs: train mode will be losses val mode will be results
        '''
        features = self.backbone(images)
        
        if self.neck:
            features = self.neck(features)
        
        outputs = self.roi_heads(features, targets)
        
        return outputs