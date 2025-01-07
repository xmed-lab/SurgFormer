import argparse
import random
import time
import numpy as np
import torch
import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F
import pickle
import torchvision.transforms.functional as TF
from sklearn import metrics


# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


# build transform
transform = T.Compose([
    T.Resize((320, 320)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_args_parser():
    parser = argparse.ArgumentParser('SurgFormer', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=3, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--query_enc_layers', default=1, type=int,
                        help="Number of query encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=4, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Loss coefficients
    parser.add_argument('--phase_loss_coef', default=1, type=float)
    parser.add_argument('--phase_ant_loss_coef', default=5, type=float)
    parser.add_argument('--tool_loss_coef', default=1, type=float)
    parser.add_argument('--tool_ant_loss_coef', default=5, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='mvp')
    parser.add_argument('--sequence_length', default=32, type=int)
    parser.add_argument('--resize', default=320, type=int)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='weights/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


def main(args):
    args.dataset_file = "mvp"
    args.batch_size == 1
    print("Inference only supports for batch size = 1")
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load data
    data_path = '/nfs/usrhome/eemenglan/71_heart/test_paths_labels1_4task_27.pkl'
    with open(data_path, 'rb') as f:
        test_paths_labels = pickle.load(f)
    test_paths = test_paths_labels[0]
    test_labels = test_paths_labels[1]
    video_list = test_paths_labels[2]
    print(video_list, sum(video_list))

    # generate the lables
    phase_label_all = [] # 12 list
    phase_ant_label_all = [] # 12 list
    tool_ant_label_all = []  # 8 list


    test_labels_12 = np.asarray(test_labels, dtype=np.float32)
    phase_labels = test_labels_12[:, 0]  # all the samples
    tool_labels = test_labels_12[:, 1: 9].astype(np.int32)
    phase_ant_labels = test_labels_12[:, 9: 21]
    tool_ant_labels = test_labels_12[:, 21: 29]


    count = 0
    for num in video_list:
        phase_label_each_video = phase_labels[count : count+num].tolist()
        phase_label_each_video = list(map(int, phase_label_each_video))
        phase_ant_label_each_video = phase_ant_labels[count : count + num].tolist()
        tool_ant_label_each_video = tool_ant_labels[count: count + num].tolist()
        count = count + num
        phase_label_all.append(phase_label_each_video)
        phase_ant_label_all.append(phase_ant_label_each_video)
        tool_ant_label_all.append(tool_ant_label_each_video)

    # model
    model, criterion = build_model(args)
    device = args.device
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        print('load weights from: ', args.resume)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
    	raise ValueError('Please specify the checkpoint for inference.')

    # start inference
    # num_all_frames = 0
    model.eval()

    # 1. for each video
    video_num = len(video_list)
    count = 0
    pred_phase_all = []
    pred_tool_all = []
    pred_phase_ant_all = []
    pred_tool_ant_all = []
    print('start inference')
    for i in range(video_num):   # video_num
        video_len = video_list[i]
        # NOTE: the im2col_step for MSDeformAttention is set as 64
        # so the max length for a clip is 64
        # store the video pred results
        pred_phase_each_video = []
        pred_tool_each_video = []
        pred_phase_ant_each_video = []
        pred_tool_ant_each_video = []

        frames_ids = [x for x in range(video_len)]
        # 3. for each clip
        for clip_id in range(0, video_len, 64):
            clip_frames_ids = frames_ids[clip_id : clip_id + 64]
            # clip_len = len(clip_frames_ids)
            # load the clip images
            imgs = []
            for t in clip_frames_ids:
                t = t+count
                img_path = test_paths[t]
                img = Image.open(img_path).convert('RGB')
                imgs.append(transform(img)) # list[Img]

            imgs = torch.stack(imgs, dim=0).to(args.device) # [video_len, 3, H, W]

            with torch.no_grad():
                time1 = time.time()
                outputs = model([imgs])
                time2 = time.time()
                time_t = time2 - time1
                print(time_t)

            pred_phase_logits = outputs["pred_phase"][0] # [t, k]
            pred_tool_logits = outputs["pred_tool"][0]  # [t, k]
            pred_phase_ant = outputs["pred_phase_ant"][0]   # [t, k]
            pred_tool_ant = outputs["pred_tool_ant"][0]  # [t, k]

            # according to pred_logits, select the query index
            pred_scores = pred_phase_logits.sigmoid() # [t, k]
            _,pred_phase = pred_scores.max(-1) # [t,]
            pred_phase = pred_phase.data.cpu()


            pred_tool_scores = pred_tool_logits.sigmoid()  # [t, k]
            pred_tool_scores = pred_tool_scores.data.cpu()
            pred_tool_scores = torch.tensor(pred_tool_scores > 0.5).int()


            pred_phase_each_video.append(pred_phase)
            pred_tool_each_video.append(pred_tool_scores)
            pred_phase_ant_each_video.append(pred_phase_ant)
            pred_tool_ant_each_video.append(pred_tool_ant)

        count = count + video_len
        pred_phase_each_video = torch.cat(pred_phase_each_video, dim=0)
        pred_tool_each_video = torch.cat(pred_tool_each_video, dim=0)
        pred_phase_ant_each_video = torch.cat(pred_phase_ant_each_video, dim=0)
        pred_tool_ant_each_video = torch.cat(pred_tool_ant_each_video, dim=0)

        pred_phase_all.append(pred_phase_each_video)
        pred_tool_all.append(pred_tool_each_video)
        pred_phase_ant_all.append(pred_phase_ant_each_video)
        pred_tool_ant_all.append(pred_tool_ant_each_video)
        print('precessed frames: ', video_len)

    # calculate the Acc, Precision, Recall and the Ja
    test_acc_each_video = []
    all_preds_phase = []
    all_labels_phase = []
    all_preds_phase_ant = []
    all_labels_phase_ant = []

    all_preds_tool = []
    all_preds_tool_ant = []
    all_labels_tool_ant = []

    video_num = len(pred_phase_all)
    for i in range(video_num):
        preds_phase = pred_phase_all[i]
        phase_label = torch.tensor(phase_label_all[i])
        frames_num = preds_phase.shape[0]
        acc = float(torch.sum(preds_phase == phase_label)) / frames_num
        test_acc_each_video.append(acc)

        preds_tool = pred_tool_all[i]


        preds_ant_phase = pred_phase_ant_all[i]
        phase_ant_label = phase_ant_label_all[i]
        #
        preds_ant_tool = pred_tool_ant_all[i]
        tool_ant_label = tool_ant_label_all[i]

        for j in range(len(preds_phase)):
            all_preds_phase.append(int(preds_phase[j]))
        for j in range(len(phase_label)):
            all_labels_phase.append(int(phase_label[j]))

        for j in range(len(preds_tool)):
            all_preds_tool.append(preds_tool.numpy()[j])
        #
        for j in range(len(preds_ant_phase)):
            all_preds_phase_ant.append(preds_ant_phase.data.cpu().numpy()[j])
        for j in range(len(phase_ant_label)):
            all_labels_phase_ant.append(phase_ant_label[j])

        for j in range(len(preds_ant_tool)):
            all_preds_tool_ant.append(preds_ant_tool.data.cpu().numpy()[j])
        for j in range(len(tool_ant_label)):
            all_labels_tool_ant.append(tool_ant_label[j])

    # phase

    print('---------------------------------------------------')
    print("Phase Recognition")
    test_acc_video = np.mean(test_acc_each_video)  # ACC
    print('Acc_video: ', test_acc_video)
    print('Acc_each_video: ', test_acc_each_video)
    test_precision_phase = metrics.precision_score(all_labels_phase, all_preds_phase, average='macro')
    print('Precision: ', test_precision_phase)
    test_recall_phase = metrics.recall_score(all_labels_phase, all_preds_phase, average='macro')
    print('Recall: ', test_recall_phase)
    test_jaccard_phase = metrics.jaccard_score(all_labels_phase, all_preds_phase, average='macro')
    print('Jaccard: ', test_jaccard_phase)
    test_precision_each_phase = metrics.precision_score(all_labels_phase, all_preds_phase, average=None)
    print('Precision_each_phase: ', test_precision_each_phase)
    test_recall_each_phase = metrics.recall_score(all_labels_phase, all_preds_phase, average=None)
    print('Recall_each_phase: ', test_recall_each_phase)
    test_jaccard_each_phase = metrics.jaccard_score(all_labels_phase, all_preds_phase, average=None)
    print('Jaccard_each_phase: ', test_jaccard_each_phase)

    print('---------------------------------------------------')
    print("Tool Recognition")
    # tool recognition
    all_preds_tool = np.array(all_preds_tool)
    test_precision_tool = metrics.average_precision_score(tool_labels, all_preds_tool, average='macro')
    print('mAP of tool recognition: ', test_precision_tool)
    test_precision_each_tool = metrics.average_precision_score(tool_labels, all_preds_tool, average=None)
    print('AP of each tool: ', test_precision_each_tool)


    # phase_anticipation
    all_preds_phase_ant = np.array(all_preds_phase_ant).transpose(1, 0)
    all_labels_phase_ant = np.array(all_labels_phase_ant).transpose(1, 0)
    horizon = 5
    in_MAE = []
    eMAE = []
    pMAE = []
    for y, t in zip(all_preds_phase_ant, all_labels_phase_ant):
        inside_horizon = (t > 0.0) & (t < 1.0)
        e_anticipating = (t < 1*.1) & (t > 0.0)
        anticipating = (y > 1 * .1) & (y < 1 * .9)

        in_MAE_ins = np.mean(np.abs(y[inside_horizon]*horizon-t[inside_horizon]*horizon))
        if not np.isnan(in_MAE_ins):
            in_MAE.append(in_MAE_ins)

        pMAE_ins = np.mean(np.abs(y[anticipating]*horizon-t[anticipating]*horizon))
        if not np.isnan(pMAE_ins):
            pMAE.append(pMAE_ins)

        eMAE_ins = np.mean(np.abs(y[e_anticipating]*horizon-t[e_anticipating]*horizon))
        if not np.isnan(eMAE_ins):
            eMAE.append(eMAE_ins)

    print('---------------------------------------------------')
    print("Phase Anticipation")
    in_MAE_phase = np.mean(in_MAE)
    print('in_MAE_test of Phase_ant: ', in_MAE_phase)
    pMAE_phase = np.mean(pMAE)
    print('pMAE_test of Phase_ant: ', pMAE_phase)
    eMAE_phase = np.mean(eMAE)
    print('eMAE_test of Phase_ant: ', eMAE_phase)


    # tool anticipation
    all_preds_tool_ant = np.array(all_preds_tool_ant).transpose(1, 0)
    all_labels_tool_ant = np.array(all_labels_tool_ant).transpose(1, 0)
    horizon = 5
    in_MAE = []
    eMAE = []
    pMAE = []
    for y, t in zip(all_preds_tool_ant, all_labels_tool_ant):
        inside_horizon = (t > 0.0) & (t < 1.0)
        e_anticipating = (t < 1 * .1) & (t > 0.0)
        anticipating = (y > 1 * .1) & (y < 1 * .9)

        in_MAE_ins = np.mean(np.abs(y[inside_horizon] * horizon - t[inside_horizon] * horizon))
        if not np.isnan(in_MAE_ins):
            in_MAE.append(in_MAE_ins)

        pMAE_ins = np.mean(np.abs(y[anticipating] * horizon - t[anticipating] * horizon))
        if not np.isnan(pMAE_ins):
            pMAE.append(pMAE_ins)

        eMAE_ins = np.mean(np.abs(y[e_anticipating] * horizon - t[e_anticipating] * horizon))
        if not np.isnan(eMAE_ins):
            eMAE.append(eMAE_ins)

    print('---------------------------------------------------')
    print("Tool Anticipation")
    in_MAE_tool = np.mean(in_MAE)
    print('in_MAE_test of Tool_ant: ', in_MAE_tool)
    pMAE_tool = np.mean(pMAE)
    print('pMAE_test of Tool_ant: ', pMAE_tool)
    eMAE_tool = np.mean(eMAE)
    print('eMAE_test of Tool_ant: ', eMAE_tool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MVP inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

