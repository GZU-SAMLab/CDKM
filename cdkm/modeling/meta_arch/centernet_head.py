import math
from typing import List
import torch
from torch import nn
from torch.nn import functional as F


__all__ = ["Proposal_Head"]

from third_party.CenterNet2.detectron2.config import configurable
from third_party.CenterNet2.detectron2.layers import get_norm


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
    
class implict_layer(nn.Module):
    def __init__(self,function,channel) -> None:
        super(implict_layer,self).__init__()
        if function == 'add':
            self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
            nn.init.normal_(self.implicit, std=.02)
        elif function == 'mul':
            self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
            nn.init.normal_(self.implicit, mean=1., std=.02)
        else:
            raise NotImplementedError
    def forward(self):
        return self.implicit
    
class Proposal_Head(nn.Module):
    @configurable
    def __init__(self, 
        # input_shape: List[ShapeSpec],
        in_channels,
        num_levels,
        *,
        num_classes=80,
        with_agn_hm=True,
        only_proposal=True,
        norm='GN',
        num_cls_convs=4,
        num_box_convs=4,
        num_share_convs=0,
        use_deformable=False,
        prior_prob=0.01,
        num_fpn_convs=2,
        use_implicit_knowledge=True):
        super().__init__()
        self.use_implicit_knowledge = use_implicit_knowledge
        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(num_levels)])
        
        tower = []
        for i in range(num_fpn_convs):
            conv_func = nn.Conv2d
            tower.append(conv_func(in_channels,in_channels,kernel_size=3, stride=1,padding=1, bias=True))
            if norm == 'GN' and in_channels % 32 != 0:
                tower.append(nn.GroupNorm(25, in_channels))
            elif norm != '':
                tower.append(get_norm(norm, in_channels))
            tower.append(nn.ReLU())
        self.add_module('fpn_shared_layer',nn.Sequential(*tower))

        if self.use_implicit_knowledge:
            self.fpn_shared_implicit_A = implict_layer('add',in_channels)
            self.fpn_shared_implicit_B = implict_layer('mul',in_channels)
        
        self.pred_heads = nn.ModuleList(
            [pred_head(
            in_channels,
            num_levels,
            num_classes,
            with_agn_hm,
            only_proposal,
            norm,
            num_cls_convs,
            num_box_convs,
            num_share_convs,
            prior_prob,
            use_deformable,
            use_implicit_knowledge=True
            ) for _ in range(num_levels)]
        )
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            # 'input_shape': input_shape,
            'in_channels': [s.channels for s in input_shape][0],
            'num_levels': len(input_shape),
            'num_classes': cfg.MODEL.CENTERNET.NUM_CLASSES,
            'with_agn_hm': cfg.MODEL.CENTERNET.WITH_AGN_HM,
            'only_proposal': cfg.MODEL.CENTERNET.ONLY_PROPOSAL,
            'norm': cfg.MODEL.CENTERNET.NORM,
            'num_cls_convs': cfg.MODEL.CENTERNET.NUM_CLS_CONVS,
            'num_box_convs': cfg.MODEL.CENTERNET.NUM_BOX_CONVS,
            'num_share_convs': cfg.MODEL.CENTERNET.NUM_SHARE_CONVS,
            'use_deformable': cfg.MODEL.CENTERNET.USE_DEFORMABLE,
            'prior_prob': cfg.MODEL.CENTERNET.PRIOR_PROB,
        }
        return ret
    
    def forward(self,x):
        clss = []
        bbox_reg = []
        agn_hms = []
        for l, feature in enumerate(x):
            feature = self.fpn_shared_layer(feature)
            if self.use_implicit_knowledge:
                feature = self.fpn_shared_implicit_A().expand_as(feature) + feature
                feature = self.fpn_shared_implicit_B().expand_as(feature) * feature
            cls,agn_hm,reg = self.pred_heads[l](feature)
            reg = self.scales[l](reg)
            bbox_reg.append(F.relu(reg))
            clss.append(cls)
            agn_hms.append(agn_hm)
        return clss, bbox_reg, agn_hms

class pred_head(nn.Module):
    def __init__(
            self,
            in_channels,
            num_levels,
            num_classes=80,
            with_agn_hm=False,
            only_proposal=False,
            norm='GN',
            num_cls_convs=4,
            num_box_convs=4,
            num_share_convs=0,
            prior_prob=0.01,
            use_deformable=False,
            use_implicit_knowledge=True
                 ) -> None:
        super(pred_head,self).__init__()
        self.num_classes = num_classes
        self.with_agn_hm = with_agn_hm
        self.only_proposal = only_proposal
        self.out_kernel = 3
        self.use_implicit_knowledge = use_implicit_knowledge

        head_configs = {
            "cls": (num_cls_convs if not self.only_proposal else 0, \
                use_deformable),
            "bbox": (num_box_convs, use_deformable),
            "share": (num_share_convs, use_deformable)}

        # in_channels = [s.channels for s in input_shape]
        # assert len(set(in_channels)) == 1, \
        #     "Each level must have the same channel!"
        # in_channels = in_channels[0]
        channels = {
            'cls': in_channels,
            'bbox': in_channels,
            'share': in_channels,
        }
        for head in head_configs:
            tower = []
            num_convs, use_deformable = head_configs[head]
            channel = channels[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                tower.append(conv_func(
                        in_channels if i == 0 else channel,
                        channel, 
                        kernel_size=3, stride=1,
                        padding=1, bias=True
                ))
                if norm == 'GN' and channel % 32 != 0:
                    tower.append(nn.GroupNorm(25, channel))
                elif norm != '':
                    tower.append(get_norm(norm, channel))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=self.out_kernel,
            stride=1, padding=self.out_kernel // 2
        )

        self.scales = nn.ModuleList(
            [Scale(init_value=1.0) for _ in range(num_levels)])

        for modules in [
            self.cls_tower, self.bbox_tower,
            self.share_tower,
            self.bbox_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        
        torch.nn.init.constant_(self.bbox_pred.bias, 8.)
        prior_prob = prior_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        if self.with_agn_hm:
            self.agn_hm = nn.Conv2d(
                in_channels, 1, kernel_size=self.out_kernel,
                stride=1, padding=self.out_kernel // 2
            )
            torch.nn.init.constant_(self.agn_hm.bias, bias_value)
            torch.nn.init.normal_(self.agn_hm.weight, std=0.01)

        if not self.only_proposal:
            cls_kernel_size = self.out_kernel
            self.cls_logits = nn.Conv2d(
                in_channels, self.num_classes,
                kernel_size=cls_kernel_size, 
                stride=1,
                padding=cls_kernel_size // 2,
            )

            torch.nn.init.constant_(self.cls_logits.bias, bias_value)
            torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        
        if self.use_implicit_knowledge:
            if not self.only_proposal:
                self.cls_implicit_A = implict_layer('add',in_channels)
                self.cls_implicit_B = implict_layer('mul',in_channels)
            
            self.bbox_implicit_A = implict_layer('add',in_channels)
            self.bbox_implicit_B = implict_layer('mul',in_channels)

    def forward(self,x):
        feature = self.share_tower(x)
        cls_tower = self.cls_tower(feature)
        bbox_tower = self.bbox_tower(feature)

        if self.use_implicit_knowledge:
            bbox_tower = self.bbox_implicit_A().expand_as(bbox_tower) + bbox_tower
            bbox_tower = self.bbox_implicit_B().expand_as(bbox_tower) * bbox_tower

        if not self.only_proposal:
            if self.use_implicit_knowledge:
                cls_tower = self.cls_implicit_A().expand_as(cls_tower) + cls_tower
                cls_tower = self.cls_implicit_B().expand_as(cls_tower) * cls_tower
            clss = self.cls_logits(cls_tower)
        else:
            clss = None
        if self.with_agn_hm:
            agn_hms = self.agn_hm(bbox_tower)
        else:
            agn_hms = None
        reg = self.bbox_pred(bbox_tower)

        return clss,agn_hms,reg




