import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_videos_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy
from einops import rearrange, repeat

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SurgFormer(nn.Module):
    def __init__(self, backbone, transformer, phase_classes, phase_ant_classes, tool_classes,tool_ant_classes, num_queries, num_feature_levels, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.phase_embed = nn.Linear(hidden_dim, phase_classes)
        self.phase_ant_embed = nn.Linear(hidden_dim, phase_ant_classes)
        self.tool_embed = nn.Linear(hidden_dim, tool_classes)
        self.tool_ant_embed = nn.Linear(hidden_dim, tool_ant_classes)

        self.num_feature_levels = num_feature_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        self.phase_embed.bias.data = torch.ones(phase_classes) * bias_value
        self.phase_ant_embed.bias.data = torch.ones(phase_ant_classes) * bias_value
        self.tool_embed.bias.data = torch.ones(tool_classes) * bias_value
        self.tool_ant_embed.bias.data = torch.ones(tool_ant_classes) * bias_value

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)


        self.phase_embed = nn.ModuleList([self.phase_embed for _ in range(num_pred)])
        self.phase_ant_embed = nn.ModuleList([self.phase_ant_embed for _ in range(num_pred)])
        self.tool_embed = nn.ModuleList([self.tool_embed for _ in range(num_pred)])
        self.tool_ant_embed = nn.ModuleList([self.tool_ant_embed for _ in range(num_pred)])

        self.transformer.decoder.bbox_embed = None

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples)
        bs = samples.tensors.shape[0]
        t = samples.tensors.shape[1]
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = self.query_embed.weight
        query_embeds = repeat(query_embeds, 'q c -> b t q c', b=bs, t=t)
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, query_embeds)

        outputs_phases = []
        outputs_phases_ant = []
        outputs_tool = []
        outputs_tool_ant = []

        for lvl in range(hs.shape[0]):
            output_phases = self.phase_embed[lvl](hs[lvl][:, 0])
            output_phases_ant = self.phase_ant_embed[lvl](hs[lvl][:, 1])
            output_tool = self.tool_embed[lvl](hs[lvl][:, 2])
            output_tool_ant = self.tool_ant_embed[lvl](hs[lvl][:, 3])

            outputs_phases.append(output_phases)
            outputs_phases_ant.append(output_phases_ant)
            outputs_tool.append(output_tool)
            outputs_tool_ant.append(output_tool_ant)

        output_phases = torch.stack(outputs_phases)
        output_phases_ant = torch.stack(outputs_phases_ant)
        output_tool = torch.stack(outputs_tool)
        output_tool_ant = torch.stack(outputs_tool_ant)


        output_phases = rearrange(output_phases, 'l (b t) k -> l b t k', b=bs, t=t)
        output_phases_ant = rearrange(output_phases_ant, 'l (b t) k -> l b t k', b=bs, t=t)
        output_tool = rearrange(output_tool, 'l (b t) k -> l b t k', b=bs, t=t)
        output_tool_ant = rearrange(output_tool_ant, 'l (b t) k -> l b t k', b=bs, t=t)

        # out = {'pred_phase_ant': output_phases_ant[-1], 'pred_tool': output_tool[-1]}
        out = {'pred_phase': output_phases[-1], 'pred_phase_ant': output_phases_ant[-1], 'pred_tool': output_tool[-1], 'pred_tool_ant': output_tool_ant[-1]}
        if self.aux_loss:
            # out['aux_outputs'] = self._set_aux_loss(output_phases_ant, output_tool)
            out['aux_outputs'] = self._set_aux_loss(output_phases, output_phases_ant, output_tool, output_tool_ant)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, output_phases, output_phases_ant, output_tool, output_tool_ant):

        return [{'pred_phase': a, 'pred_phase_ant': b, 'pred_tool': c, 'pred_tool_ant': d}
            for a, b, c, d in zip(output_phases[:-1], output_phases_ant[:-1], output_tool[:-1], output_tool_ant[:-1])]


class SetCriterion(nn.Module):
    def __init__(self, weight_dict, losses, focal_alpha=0.25):
        super().__init__()
        # self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.L1 = nn.SmoothL1Loss()
        self.bce_logit = nn.BCEWithLogitsLoss()

    def loss_labels(self, outputs, targets):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_phase' in outputs
        src_logits = outputs['pred_phase']
        phase_tmp = []
        for target in targets:
            tmp = target['labels_phase']
            phase_tmp.append(tmp)
        target_classes_onehot = torch.stack(phase_tmp)

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_phase': loss_ce}

        return losses

    def loss_phase_ant(self, outputs, targets):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_phase_ant' in outputs
        src_logits = outputs['pred_phase_ant']
        phase_ant_tmp = []
        for target in targets:
            tmp = target['labels_phase_ant']
            phase_ant_tmp.append(tmp)
        target_ant = torch.stack(phase_ant_tmp)

        loss_l1 = self.L1(src_logits, target_ant) * src_logits.shape[1]
        losses = {'loss_phase_ant': loss_l1}

        return losses
    def loss_tool(self, outputs, targets):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_tool' in outputs
        src_logits = outputs['pred_tool']
        tool_tmp = []
        for target in targets:
            tmp = target['labels_tool']
            tool_tmp.append(tmp)
        target_tool = torch.stack(tool_tmp)
        loss_bce = self.bce_logit(src_logits, target_tool) * src_logits.shape[1]
        losses = {'loss_tool': loss_bce}

        return losses

    def loss_tool_ant(self, outputs, targets):

        assert 'pred_tool_ant' in outputs
        src_logits = outputs['pred_tool_ant']
        tool_ant_tmp = []
        for target in targets:
            tmp = target['labels_tool_ant']
            tool_ant_tmp.append(tmp)
        target_ant = torch.stack(tool_ant_tmp)
        loss_l1 = self.L1(src_logits, target_ant) * src_logits.shape[1]
        losses = {'loss_tool_ant': loss_l1}

        return losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'phase': self.loss_labels,
            'phase_ant': self.loss_phase_ant,
            'tool': self.loss_tool,
            'tool_ant': self.loss_tool_ant,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    if args.dataset_file == 'mvp':
        phase_classes = 12
        phase_ant_classes = 12
        tool_classes = 8
        tool_ant_classes = 8
    elif args.dataset_file == 'cholec80':
        phase_classes = 7
        phase_ant_classes = 7
        tool_classes = 7
        tool_ant_classes = 7
    else:
        phase_classes = 7
        phase_ant_classes = 7
        tool_classes = 7
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = SurgFormer(
        backbone,
        transformer,
        phase_classes=phase_classes,
        phase_ant_classes=phase_ant_classes,
        tool_classes=tool_classes,
        tool_ant_classes=tool_ant_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss
    )

    weight_dict = {'loss_phase': args.phase_loss_coef, 'loss_phase_ant': args.phase_ant_loss_coef, 'loss_tool': args.tool_loss_coef, 'loss_tool_ant': args.tool_ant_loss_coef}

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['phase', 'phase_ant', 'tool', 'tool_ant']
    criterion = SetCriterion(weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)

    return model, criterion
