import math
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
from timm.models.layers import Mlp

from detectron2.config import configurable
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads, _ScaleGradient
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import batched_nms
from .grit_fast_rcnn import GRiTFastRCNNOutputLayers

from ..text.text_decoder import TransformerDecoderTextualHead, GRiTTextDecoder, AutoRegressiveBeamSearch
from ..text.load_text_token import LoadTextTokens
from transformers import BertTokenizer
from grit.data.custom_dataset_mapper import ObjDescription
from ..soft_nms import batched_soft_nms

import logging
logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class GRiTROIHeadsAndTextDecoder(CascadeROIHeads):
    @configurable
    def __init__(
        self,
        *,
        text_decoder_transformer,
        train_task: list,
        test_task: str,  # 要进行测试的任务名称
        mult_proposal_score: bool = False,  # 是否使用多目标提案分数
        mask_weight: float = 1.0,  # 掩码权重
        object_feat_pooler=None,  # 对象特征池化
        soft_nms_enabled=False,  # 是否启用软NMS
        beam_size=1,  # 束搜索的宽度
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mult_proposal_score = mult_proposal_score
        self.mask_weight = mask_weight
        self.object_feat_pooler = object_feat_pooler
        self.soft_nms_enabled = soft_nms_enabled
        self.test_task = test_task
        self.beam_size = beam_size

        # 创建BertTokenizer对象，加载预训练模型bert-base-uncased的分词器
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenizer = tokenizer

        # 检查test_task是否存在于train_task中，如果不存在则抛出异常
        assert test_task in train_task, 'GRiT has not been trained on {} task, ' \
                                        'please verify the task name or train a new ' \
                                        'GRiT on {} task'.format(test_task, test_task)
        task_begin_tokens = {}  # 空字典，用于存储每个任务的起始ID
        for i, task in enumerate(train_task):
            if i == 0:  # 当前第一个任务
                # 将该任务的起始标记ID赋给字典中对应的键
                task_begin_tokens[task] = tokenizer.cls_token_id
            else:
                # 将该任务的起始标记ID赋给字典中对应的键
                task_begin_tokens[task] = 103 + i
        self.task_begin_tokens = task_begin_tokens

        # 创建实例，用于执行束搜索
        beamsearch_decode = AutoRegressiveBeamSearch(
            # 设置束搜索的结束标记ID为分词器的分隔符标记ID
            end_token_id=tokenizer.sep_token_id,
            max_steps=40,  # 最大步数
            beam_size=beam_size,  # 设置束搜索的宽度
            objectdet=test_task == "ObjectDet",  # 根据test_task的值判断是否启用对象检测功能
            per_node_beam_size=1,  # 设置每个节点的束搜索宽度为1
        )

        # 创建实例，用于文本解析
        self.text_decoder = GRiTTextDecoder(
            text_decoder_transformer,
            beamsearch_decode=beamsearch_decode,  # 设置束搜索解码器
            begin_token_id=task_begin_tokens[test_task],
            loss_type='smooth',  # 设置损失类型为平滑损失
            tokenizer=tokenizer,  # 分词器
        )

        # 加载目标文本的分词器
        self.get_target_text_tokens = LoadTextTokens(tokenizer, max_text_len=40, padding='do_not_pad')
        # 区域注意力机制
        self.reg_attn = region_attention()

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        text_decoder_transformer = TransformerDecoderTextualHead(
            object_feature_size=cfg.MODEL.FPN.OUT_CHANNELS,
            vocab_size=cfg.TEXT_DECODER.VOCAB_SIZE,
            hidden_size=cfg.TEXT_DECODER.HIDDEN_SIZE,
            num_layers=cfg.TEXT_DECODER.NUM_LAYERS,
            attention_heads=cfg.TEXT_DECODER.ATTENTION_HEADS,
            feedforward_size=cfg.TEXT_DECODER.FEEDFORWARD_SIZE,
            mask_future_positions=True,
            padding_idx=0,
            decoder_type='bert_en',
            use_act_checkpoint=cfg.USE_ACT_CHECKPOINT,
        )
        ret.update({
            'text_decoder_transformer': text_decoder_transformer,
            'train_task': cfg.MODEL.TRAIN_TASK,
            'test_task': cfg.MODEL.TEST_TASK,
            'mult_proposal_score': cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE,
            'mask_weight': cfg.MODEL.ROI_HEADS.MASK_WEIGHT,
            'soft_nms_enabled': cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED,
            'beam_size': cfg.MODEL.BEAM_SIZE,
        })
        return ret

    # 根据配置文件中的信息构建一个区域提议网络的框头部
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        # 删除返回结果中的'box_predictors'键及其对应的值
        del ret['box_predictors']
        # 获取级联边界框回归权重
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []  # 创建一个空列表用于存储框预测器
        for box_head, bbox_reg_weights in zip(ret['box_heads'], \
            cascade_bbox_reg_weights):
            # 将每个框头部的输出层添加到box_predictors列表中
            box_predictors.append(
                GRiTFastRCNNOutputLayers(
                    cfg, box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights)
                ))
        # 将构建好的box_predictors列表赋值给返回结果的'box_predictors'键
        ret['box_predictors'] = box_predictors

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES  # 获取输入特征数
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)  # 计算池化比例
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO  # 获取采样比率
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE  # 获取池化类型
        # 创建一个区域池化器对象
        object_feat_pooler = ROIPooler(
            output_size=cfg.MODEL.ROI_HEADS.OBJECT_FEAT_POOLER_RES,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # 将构建好的对象特征池化器赋值给返回结果的'object_feat_pooler'键
        ret['object_feat_pooler'] = object_feat_pooler
        return ret

    def check_if_all_background(self, proposals, targets, stage):
        all_background = True  # 初始化为所有候选框都是背景
        for proposals_per_image in proposals:  # 遍历每个图像的候选框
            if not (proposals_per_image.gt_classes == self.num_classes).all():
                all_background = False  # 存在非背景框

        if all_background:  # 如果所有候选框都是背景
            logger.info('all proposals are background at stage {}'.format(stage))  # 记录日志信息
            proposals[0].proposal_boxes.tensor[0, :] = targets[0].gt_boxes.tensor[0, :]  # 更新第一个候选框的预测框
            proposals[0].gt_boxes.tensor[0, :] = targets[0].gt_boxes.tensor[0, :]  # 更新第一个候选框的真实框
            proposals[0].objectness_logits[0] = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))  # 更新第一个候选框的置信度
            proposals[0].gt_classes[0] = targets[0].gt_classes[0]  # 更新第一个候选框的类别
            proposals[0].gt_object_descriptions.data[0] = targets[0].gt_object_descriptions.data[0]  # 更新第一个候选框的物体描述
            if 'foreground' in proposals[0].get_fields().keys():  # 如果第一个候选框是前景
                proposals[0].foreground[0] = 1  # 将前景标记设置为1
        return proposals

    def _forward_box(self, features, proposals, targets=None, task="ObjectDet"):
        # 如果处于训练模式，检查是否所有候选框都是背景
        if self.training:
            proposals = self.check_if_all_background(proposals, targets, 0)
        # 如果处于非训练（测试）模式且需要计算多个建议框的分数
        if (not self.training) and self.mult_proposal_score:
            # 如果第一个候选框有分数
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [p.get('scores') for p in proposals]
            else:
                proposal_scores = [p.get('objectness_logits') for p in proposals]

        features = [features[f] for f in self.box_in_features]  # 获取特征中的边界框特征
        head_outputs = []  # 初始化头输出列表
        prev_pred_boxes = None  # 初始化上一个预测框
        image_sizes = [x.image_size for x in proposals]  # 获取图像尺寸

        # 遍历每个级联阶段
        for k in range(self.num_cascade_stages):
            # 如果不是第一个阶段
            if k > 0:
                proposals = self._create_proposals_from_boxes(   # 根据之前的预测框和图像尺寸创建新的建议框
                    prev_pred_boxes, image_sizes,
                    logits=[p.objectness_logits for p in proposals])
                if self.training:  # 如果处于训练模式，匹配并标记框
                    proposals = self._match_and_label_boxes_GRiT(
                        proposals, k, targets)
                    proposals = self.check_if_all_background(proposals, targets, k)
            predictions = self._run_stage(features, proposals, k)  # 预测结果
            # 更新上一个预测框
            prev_pred_boxes = self.box_predictor[k].predict_boxes((predictions[0], predictions[1]), proposals)
            # 将预测结果添加到头输出列表中
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            object_features = self.object_feat_pooler(features, [x.proposal_boxes for x in proposals])  # 获取目标特征
            object_features = _ScaleGradient.apply(object_features, 1.0 / self.num_cascade_stages)  # 对目标特征进行缩放
            reg_tokens = self.get_region_attention(object_features, proposals)  # 获取区域注意力
            foreground = torch.cat([x.foreground for x in proposals])  # 获取前景框
            object_features = object_features[foreground > 0]  # 获取前景框对应的目标特征
            reg_tokens = reg_tokens[foreground > 0]  # 获取正则化令牌

            object_descriptions = []  # 初始化对象描述列表
            # 遍历建议框，获取目标特征和对象描述
            for x in proposals:
                object_descriptions += x.gt_object_descriptions[x.foreground > 0].data
            # 将对象描述转换为张量
            object_descriptions = ObjDescription(object_descriptions)
            object_descriptions = object_descriptions.data

            # 如果对象描述不为空
            if len(object_descriptions) > 0:
                # 获取开始令牌
                begin_token = self.task_begin_tokens[task]
                # 获取文本解码器输入
                text_decoder_inputs = self.get_target_text_tokens(object_descriptions, object_features, begin_token)
                # 更新对象特征
                object_features = object_features.view(
                    object_features.shape[0], object_features.shape[1], -1).permute(0, 2, 1).contiguous()
                # 将正则化令牌和对象特征拼接
                object_features = torch.cat([reg_tokens,object_features],dim=1)
                # 更新文本解码器输入
                text_decoder_inputs.update({'object_features': object_features})
                # 计算文本解码器损失
                text_decoder_loss = self.text_decoder(text_decoder_inputs)
            else:
                # 如果对象描述为空，设置文本解码器损失为0
                text_decoder_loss = head_outputs[0][1][0].new_zeros([1])[0]

            losses = {}  # 初始化损失字典
            storage = get_event_storage()  # 获取事件存储
            # RoI Head losses (For the proposal generator loss, please find it in grit.py)
            # 计算提议生成器损失（请在 grit.py 中找到）
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                # 更新损失
                with storage.name_scope("stage{}".format(stage)):
                        stage_losses = predictor.losses(
                            (predictions[0], predictions[1]), proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            # Text Decoder loss
            losses.update({'text_decoder_loss': text_decoder_loss})
            return losses
        else:
            # 计算每个阶段的概率预测结果
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            logits_per_stage = [(h[1][0],) for h in head_outputs]
            # 对每个阶段的预测结果进行平均
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            logits = [
                sum(list(logits_per_image)) * (1.0 / self.num_cascade_stages)
                for logits_per_image in zip(*logits_per_stage)
            ]
            # 如果需要计算提议分数，则将得分与提议分数相乘
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 for s, ps in zip(scores, proposal_scores)]

            # 获取最后一个头输出的预测器、预测结果和建议框
            predictor, predictions, proposals = head_outputs[-1]
            # 使用预测器预测建议框的位置
            boxes = predictor.predict_boxes(
                (predictions[0], predictions[1]), proposals)
            assert len(boxes) == 1

            # 对预测结果进行Faster R-CNN 推理
            pred_instances, _ = self.fast_rcnn_inference_GRiT(
                boxes,
                scores,
                logits,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
                self.soft_nms_enabled,
            )

            # 确保只有一个预测结果
            assert len(pred_instances) == 1, "Only support one image"

            # 遍历预测结果
            for i, pred_instance in enumerate(pred_instances):
                # 如果预测框不为空
                if len(pred_instance.pred_boxes) > 0:
                    # 提取特征并将其传递给区域注意力模块
                    object_features = self.object_feat_pooler(features, [pred_instance.pred_boxes])
                    reg_tokens = self.get_region_attention(object_features,pred_instances)
                    object_features = object_features.view(
                        object_features.shape[0], object_features.shape[1], -1).permute(0, 2, 1).contiguous()
                    object_features = torch.cat([reg_tokens,object_features],dim=1)
                    # 使用文本解码器生成文本描述
                    text_decoder_output = self.text_decoder({'object_features': object_features})
                    # 如果 beam_size 大于 1 且测试任务为 ObjectDet
                    if self.beam_size > 1 and self.test_task == "ObjectDet":
                        # 初始化预测框、得分和类别列表
                        pred_boxes = []
                        pred_scores = []
                        pred_classes = []
                        pred_object_descriptions = []

                        # 对每个 beam_id 进行循环
                        for beam_id in range(self.beam_size):
                            # 将预测框添加到列表中
                            pred_boxes.append(pred_instance.pred_boxes.tensor)

                            # object score = sqrt(objectness score x description score)
                            # 计算对象得分（对象性得分 * 描述得分）
                            pred_scores.append((pred_instance.scores *
                                                torch.exp(text_decoder_output['logprobs'])[:, beam_id]) ** 0.5)
                            # 添加预测类别
                            pred_classes.append(pred_instance.pred_classes)

                            # 遍历文本解码器的预测结果
                            for prediction in text_decoder_output['predictions'][:, beam_id, :]:
                                # convert text tokens to words 将文本标记转换为单词
                                description = self.tokenizer.decode(prediction.tolist()[1:], skip_special_tokens=True)
                                pred_object_descriptions.append(description)

                        # 合并实例
                        merged_instances = Instances(image_sizes[0])
                        if torch.cat(pred_scores, dim=0).shape[0] <= predictor.test_topk_per_image:
                            merged_instances.scores = torch.cat(pred_scores, dim=0)
                            merged_instances.pred_boxes = Boxes(torch.cat(pred_boxes, dim=0))
                            merged_instances.pred_classes = torch.cat(pred_classes, dim=0)
                            merged_instances.pred_object_descriptions = ObjDescription(pred_object_descriptions)
                        else:
                            # 获取 top k 的预测分数和索引
                            pred_scores, top_idx = torch.topk(
                                torch.cat(pred_scores, dim=0), predictor.test_topk_per_image)
                            # 更新合并实例的预测分数、预测框、类别和对象描述
                            merged_instances.scores = pred_scores
                            merged_instances.pred_boxes = Boxes(torch.cat(pred_boxes, dim=0)[top_idx, :])
                            merged_instances.pred_classes = torch.cat(pred_classes, dim=0)[top_idx]
                            merged_instances.pred_object_descriptions = \
                                ObjDescription(ObjDescription(pred_object_descriptions)[top_idx].data)

                        pred_instances[i] = merged_instances
                    # 否则，直接更新预测实例的得分和对象描述
                    else:
                        # object score = sqrt(objectness score x description score) 对象得分
                        pred_instance.scores = (pred_instance.scores *
                                                torch.exp(text_decoder_output['logprobs'])) ** 0.5

                        pred_object_descriptions = []
                        for prediction in text_decoder_output['predictions']:
                            # convert text tokens to words 将文本标记转换为单词
                            description = self.tokenizer.decode(prediction.tolist()[1:], skip_special_tokens=True)
                            pred_object_descriptions.append(description)
                        pred_instance.pred_object_descriptions = ObjDescription(pred_object_descriptions)
                # 如果预测框为空，则清空对象描述列表
                else:
                    pred_instance.pred_object_descriptions = ObjDescription([])

            return pred_instances  # 返回预测实例列表


    def forward(self, features, proposals, targets=None, targets_task="ObjectDet"):
        if self.training:
            proposals = self.label_and_sample_proposals(  # 对proposals进行标签和采样
                proposals, targets)  # 传入proposals和targets

            # 计算box损失
            losses = self._forward_box(features, proposals, targets, task=targets_task)
            if targets[0].has('gt_masks'):  # 如果目标有ground truth掩码
                mask_losses = self._forward_mask(features, proposals)  # 计算掩码损失
                # 更新损失
                losses.update({k: v * self.mask_weight \
                    for k, v in mask_losses.items()})
            else:
                losses.update(self._get_empty_mask_loss(device=proposals[0].objectness_logits.device))
            return proposals, losses

        else:
            pred_instances = self._forward_box(features, proposals, task=self.test_task)
            # 根据给定的boxes进行预测
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}  # 返回预测结果和一个空字典

    @torch.no_grad()
    def _match_and_label_boxes_GRiT(self, proposals, stage, targets):
        """
        Add  "gt_object_description" and "foreground" to detectron2's _match_and_label_boxes
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                foreground = torch.ones_like(gt_classes)
                foreground[proposal_labels == 0] = 0
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
                gt_object_descriptions = targets_per_image.gt_object_descriptions[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                foreground = torch.zeros_like(gt_classes)
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
                gt_object_descriptions = ObjDescription(['None' for i in range(len(proposals_per_image))])
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes
            proposals_per_image.gt_object_descriptions = gt_object_descriptions
            proposals_per_image.foreground = foreground

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
            )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
            )
        return proposals

    def fast_rcnn_inference_GRiT(
            self,
            boxes: List[torch.Tensor],
            scores: List[torch.Tensor],
            logits: List[torch.Tensor],
            image_shapes: List[Tuple[int, int]],
            score_thresh: float,
            nms_thresh: float,
            topk_per_image: int,
            soft_nms_enabled: bool,
    ):
        result_per_image = [
            self.fast_rcnn_inference_single_image_GRiT(
                boxes_per_image, scores_per_image, logits_per_image, image_shape,
                score_thresh, nms_thresh, topk_per_image, soft_nms_enabled
            )
            for scores_per_image, boxes_per_image, image_shape, logits_per_image \
            in zip(scores, boxes, image_shapes, logits)
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image]

    def fast_rcnn_inference_single_image_GRiT(
            self,
            boxes,
            scores,
            logits,
            image_shape: Tuple[int, int],
            score_thresh: float,
            nms_thresh: float,
            topk_per_image: int,
            soft_nms_enabled,
    ):
        """
        Add soft NMS to detectron2's fast_rcnn_inference_single_image
        """
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            logits = logits[valid_mask]

        scores = scores[:, :-1]
        logits = logits[:, :-1]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

        # 1. Filter results based on detection scores. It can make NMS more efficient
        #    by filtering out low-confidence detections.
        filter_mask = scores > score_thresh  # R x K
        # R' x 2. First column contains indices of the R predictions;
        # Second column contains indices of classes.
        filter_inds = filter_mask.nonzero()
        if num_bbox_reg_classes == 1:
            boxes = boxes[filter_inds[:, 0], 0]
        else:
            boxes = boxes[filter_mask]
        scores = scores[filter_mask]
        logits = logits[filter_mask]

        # 2. Apply NMS for each class independently.
        if not soft_nms_enabled:
            keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
        else:
            keep, soft_nms_scores = batched_soft_nms(
                boxes,
                scores,
                filter_inds[:, 1],
                "linear",
                0.5,
                nms_thresh,
                0.001,
            )
            scores[keep] = soft_nms_scores
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]
        logits = logits[keep]

        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = filter_inds[:, 1]
        result.logits = logits
        return result, filter_inds[:, 0]

    def _get_empty_mask_loss(self, device):
        if self.mask_on:
            return {'loss_mask': torch.zeros(
                (1, ), device=device, dtype=torch.float32)[0]}
        else:
            return {}

    def _create_proposals_from_boxes(self, boxes, image_sizes, logits):
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size, logit in zip(
            boxes, image_sizes, logits):
            boxes_per_image.clip(image_size)
            if self.training:
                inds = boxes_per_image.nonempty()
                boxes_per_image = boxes_per_image[inds]
                logit = logit[inds]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            prop.objectness_logits = logit
            proposals.append(prop)
        return proposals

    def _run_stage(self, features, proposals, stage):
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        return self.box_predictor[stage](box_features)
    
    def get_region_attention(self,object_features,proposals):
        if self.training:
            tmp = []
            start = 0
            for each in proposals:
                length = len(each.proposal_boxes)
                tmp.append(object_features[start:start+length])
                start+=length

            outputs = []
            for each in tmp:
                output = self.reg_attn(each)
                outputs.append(output)
            
            return torch.cat(outputs)
        
        else:
            return self.reg_attn(object_features)


    

class region_attention(nn.Module):
    def __init__(self,num_heads=8,dim=784,qkv_bias=True,out_dim=256,mlp_ratio=2):
        super(region_attention,self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.conv = nn.Conv2d(in_channels=256,out_channels=16,kernel_size=3,stride=2,padding=1)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.mlp = Mlp(in_features=out_dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)

    def forward(self,in_feature):
        x = self.conv(in_feature)
        N_region,channels,H,W = x.shape
        x = x.reshape(N_region,channels,H*W).permute(0,2,1)
        x = x.reshape(N_region,H*W*channels).unsqueeze(0)

        shortcut = x
        B, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(B, seq_len, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # [3,B,self.num_heads,seq_len,head_dim]
        q, k, v = qkv.reshape(3, B * self.num_heads, seq_len, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, seq_len, -1).permute(0, 2, 1, 3).reshape(B, seq_len, -1)
        x = shortcut + x
        x = self.norm1(self.proj(x))
        x = x + self.mlp(x)

        return x.squeeze(0).unsqueeze(1)
