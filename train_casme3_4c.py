import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from models.Exp_CLIP import VQCodeTransformer, PretrainedTextEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
import warnings
import cv2
import pandas as pd
# from models.Text import *
import gc
from PIL import Image
from sklearn.metrics import recall_score, f1_score
from einops import rearrange
from torchvision.utils import make_grid, save_image
from laq_model import LatentActionQuantization
from torchvision import transforms as T

import math
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


warnings.filterwarnings("ignore", category=UserWarning)
import random
# from models.clip import clip
# from models.BLIP2_T5 import *
from models.Text import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--batch-size-test-image', type=int, default=16)
parser.add_argument('--batch-size-test-video', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--milestones', nargs='+', type=int, default=30)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--job-id', type=str, default="OK")
parser.add_argument('--laq-checkpoint', type=str, default="vae.736000.pt") # /home/yicheng/LAPA/laq/results3/vae.22000.pt 32VOX/736000
parser.add_argument('--instruction', type=str, default="Please play the role of a facial micro action describer. Objectively describe the subtle facial actions of the person present in the difference of two facial images.")
parser.add_argument('--load-model', type=str, default="CLIP_L14")
args = parser.parse_args()

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)

now = datetime.datetime.now()
train_time = now.strftime("%y-%m-%d %H:%M")
print("Training date: ", train_time)
job_id = args.job_id

print('************************')
for k, v in vars(args).items():
    print(k,'=',v)
print('************************')


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, num_loso):
        self.phase = phase
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        self.raf_path = raf_path

        # DATASET_COLUMN = 0
        SUBJECT_COLUMN =0
        NAME_COLUMN = 1
        ONSET_COLUMN = 2
        APEX_COLUMN = 3
        OFF_COLUMN = 4
        LABEL_AU_COLUMN = 5
        LABEL_ALL_COLUMN = 7
        # LABEL_3CLASS_COLUMN = 8

        df = pd.read_excel('cas(me)3_part_A_ME_label_JpgIndex_v1.xls')

        loso_dataset = num_loso.split('/')[0]
        loso_subject = num_loso.split('/')[1]

        # 强制把第一列、第二列都转换为字符串
        col0_str = df.iloc[:, 0].astype(str)
        col1_str = df.iloc[:, 1].astype(str)

        if phase == 'train':
            # mask = (col0_str != loso_dataset) | (col1_str != loso_subject)
            mask = col0_str != loso_subject
            dataset = df[mask].reset_index(drop=True)
        else:
            mask = col0_str == loso_subject
            dataset = df[mask].reset_index(drop=True)

        # Dataset = dataset.iloc[:, DATASET_COLUMN].values
        Subject = dataset.iloc[:, SUBJECT_COLUMN].values
        File_names = dataset.iloc[:, NAME_COLUMN].values
        Label_all = dataset.iloc[:, LABEL_ALL_COLUMN].values  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        Onset_num = dataset.iloc[:, ONSET_COLUMN].values
        Apex_num = dataset.iloc[:, APEX_COLUMN].values
        Offset_num = dataset.iloc[:, OFF_COLUMN].values
        Label_au = dataset.iloc[:, LABEL_AU_COLUMN].values
        # Label_3class = dataset.iloc[:, LABEL_3CLASS_COLUMN].values

        self.file_paths_on = []
        self.file_paths_off = []
        self.file_paths_apex = []
        self.label_all = []
        self.label_au = []
        self.sub= []
        self.file_names = []

        # use aligned images for training/testing
        for (f, sub, onset, apex, offset, label_all, label_au) in zip(File_names, Subject, Onset_num, Apex_num, Offset_num, Label_all, Label_au):
            if int(onset) == 0:
                continue
            if label_all == 'happy':
                self.label_all.append(0)
            elif label_all == 'surprise':
                self.label_all.append(1)
            elif label_all == 'disgust' or label_all == 'anger' or label_all == 'fear' or label_all == 'sad':
                self.label_all.append(2)
            else:
                self.label_all.append(3)
            if onset == apex:
                apex = (int(onset) + int(offset))//2
            self.file_paths_on.append(onset)
            self.file_paths_off.append(offset)
            self.file_paths_apex.append(apex)
            self.sub.append(sub)
            self.file_names.append(f)

            self.label_au.append(label_au)


    def __len__(self):
        return len(self.file_paths_on)

    def __getitem__(self, idx):
        ##sampling strategy for training set
        if self.phase == 'train':
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            on0 = str(int(onset))
            apex0 = str(int(apex))

            sub = str(self.sub[idx])
            f = str(self.file_names[idx])
        else:##sampling strategy for testing set
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]

            on0 = str(onset)
            apex0 = str(apex)

            sub = str(self.sub[idx])
            f = str(self.file_names[idx])

        label_all = self.label_all[idx]


        on0 = on0 + '.jpg'
        apex0 = apex0 + '.jpg'
        path_on0 = os.path.join(self.raf_path, sub, f, on0)
        path_apex0 = os.path.join(self.raf_path, sub, f, apex0)

        img = Image.open(path_on0)
        next_img = Image.open(path_apex0)

        transform_img = self.transform(img).unsqueeze(1)
        next_transform_img = self.transform(next_img).unsqueeze(1)

        cat_img = torch.cat([transform_img, next_transform_img], dim=1)
        return cat_img, label_all


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # alpha 可以是 None，或 Tensor [num_classes]
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, C] - logits
        targets: [B] - int64 class indices
        """
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)  # [B, C]
        targets_onehot = F.one_hot(targets, num_classes=inputs.shape[1]).float()

        pt = (probs * targets_onehot).sum(dim=1)  # [B]
        log_pt = (log_probs * targets_onehot).sum(dim=1)  # [B]

        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha).to(inputs.device)
            alpha_t = alpha_t[targets]  # [B]
            loss = -alpha_t * (1 - pt) ** self.gamma * log_pt
        else:
            loss = -(1 - pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def main():

    log_txt_path = './log/' + job_id + '-log.txt'
    # log_curve_path = './log/' + job_id + '-log.png'
    checkpoint_path = './checkpoint/' + job_id
    recorder = RecorderMeter(args.epochs)

    LOSO = ['CASME3/spNO.1', 'CASME3/spNO.10', 'CASME3/spNO.11', 'CASME3/spNO.12', 'CASME3/spNO.13', 'CASME3/spNO.138',
            'CASME3/spNO.139', 'CASME3/spNO.14', 'CASME3/spNO.142', 'CASME3/spNO.143', 'CASME3/spNO.144',
            'CASME3/spNO.145', 'CASME3/spNO.146', 'CASME3/spNO.147', 'CASME3/spNO.148', 'CASME3/spNO.149',
            'CASME3/spNO.15', 'CASME3/spNO.150', 'CASME3/spNO.151', 'CASME3/spNO.152', 'CASME3/spNO.153',
            'CASME3/spNO.154', 'CASME3/spNO.155', 'CASME3/spNO.156', 'CASME3/spNO.157', 'CASME3/spNO.158',
            'CASME3/spNO.159', 'CASME3/spNO.160', 'CASME3/spNO.161', 'CASME3/spNO.162', 'CASME3/spNO.163',
            'CASME3/spNO.165', 'CASME3/spNO.166', 'CASME3/spNO.167', 'CASME3/spNO.168', 'CASME3/spNO.169',
            'CASME3/spNO.17', 'CASME3/spNO.170', 'CASME3/spNO.171', 'CASME3/spNO.172', 'CASME3/spNO.173',
            'CASME3/spNO.174', 'CASME3/spNO.175', 'CASME3/spNO.176', 'CASME3/spNO.177', 'CASME3/spNO.178',
            'CASME3/spNO.179', 'CASME3/spNO.180', 'CASME3/spNO.181', 'CASME3/spNO.182', 'CASME3/spNO.183',
            'CASME3/spNO.184', 'CASME3/spNO.185', 'CASME3/spNO.186', 'CASME3/spNO.187', 'CASME3/spNO.188',
            'CASME3/spNO.189', 'CASME3/spNO.190', 'CASME3/spNO.192', 'CASME3/spNO.193', 'CASME3/spNO.194',
            'CASME3/spNO.195', 'CASME3/spNO.196', 'CASME3/spNO.197', 'CASME3/spNO.198', 'CASME3/spNO.2',
            'CASME3/spNO.200', 'CASME3/spNO.201', 'CASME3/spNO.202', 'CASME3/spNO.203', 'CASME3/spNO.204',
            'CASME3/spNO.206', 'CASME3/spNO.207', 'CASME3/spNO.208', 'CASME3/spNO.209', 'CASME3/spNO.210',
            'CASME3/spNO.211', 'CASME3/spNO.212', 'CASME3/spNO.213', 'CASME3/spNO.214', 'CASME3/spNO.215',
            'CASME3/spNO.216', 'CASME3/spNO.217', 'CASME3/spNO.3', 'CASME3/spNO.39', 'CASME3/spNO.4',
            'CASME3/spNO.40', 'CASME3/spNO.41', 'CASME3/spNO.42', 'CASME3/spNO.5', 'CASME3/spNO.6',
            'CASME3/spNO.7', 'CASME3/spNO.77', 'CASME3/spNO.8', 'CASME3/spNO.9']

    best_correct_all = 0
    samples_all = 0

    for n_subName in LOSO:
        print('Subject:', n_subName)

        # load prompt embedding
        FER_prompt_ = FER_prompt_list[0]
        prompt = FER_prompt_["CASME3_4"]
        pretrainedtextencoder = PretrainedTextEncoder(args=args).cuda()
        pretrainedtextencoder.eval()
        prompt_embedding = pretrainedtextencoder(prompt)

        train_dataset = RafDataSet('datasets/CASME3/', phase='train', num_loso=n_subName)
        val_dataset = RafDataSet('datasets/CASME3/', phase='test', num_loso=n_subName)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=16,
                                                   num_workers=2,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=False)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=36,
                                                 num_workers=2,
                                                 shuffle=False,
                                                 pin_memory=True)
        print('num_sub', n_subName)
        print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())

        if len(val_loader.dataset) == 0:
            continue

        # create model and load pre_trained parameters
        model = VQCodeTransformer(args,
                                  classnum=4,
                                  depth=2,
                                  seq_len=16,
                                  dropout=0.2,
                                  dim=512,
                                  mlp_dim=512,
                                  heads=8)

        laq = LatentActionQuantization(
            dim=512,
            quant_dim=32,
            codebook_size=32,
            image_size=256,
            patch_size=16,
            spatial_depth=6,
            temporal_depth=6,
            dim_head=64,
            heads=8,
            code_seq_len=16,
        ).cuda()

        laq.load(args.laq_checkpoint)

        model = model.cuda()

        # define loss function (criterion)
        class_weights = torch.tensor([3.84, 1.16, 0.46, 1.39])
        criterion = nn.CrossEntropyLoss(weight=class_weights.cuda()).cuda()

        optimizer_laq = torch.optim.AdamW(laq.parameters(), lr=2e-4, betas=(0.85, 0.98), weight_decay=1e-5)
        optimizer_model = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.85, 0.98), weight_decay=1e-5)

        best_correct = -1
        best_all_predicted, best_all_targets = None, None
        best_uf1, best_uar = 0.0, 0.0


        for epoch in range(0, args.epochs):

            inf = '********************' + str(epoch) + '********************'
            start_time = time.time()
            print(inf)

            # train for one epoch
            train_acc, train_los_cls, train_los_con = train(train_loader, laq, model, prompt_embedding, criterion, optimizer_laq, optimizer_model, epoch, args, log_txt_path)

            # print and save log
            epoch_time = time.time() - start_time
            recorder.update(epoch, train_los_cls, train_los_con, train_acc)
            print('The train accuracy: {:.3f}'.format(train_acc))
            print('An epoch time: {:.2f}s'.format(epoch_time))

            correct1, correct2, correct3, correct4 = 0, 0, 0, 0

            model.eval()
            laq.eval()

            # save model
            state_dict = model.state_dict()
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
            }
            checkpoint_name = checkpoint_path + n_subName.split('/')[1] + '_' + str(epoch) + '.pth'
            torch.save({
                'model': filtered_state_dict,
                'laq': laq.state_dict(),
            }, checkpoint_name)

            # evaluation
            with torch.no_grad():
                for i, (images, label_all) in enumerate(val_loader):

                    images = images.cuda()

                    loss_rec, num_unique_indices, tokens4alignment, indices, returned_recon, returned_recon4inf = laq(
                        images,
                    )

                    # save reconstrcuted images
                    imgs_and_recons = torch.stack((images[:, :, 0], images[:, :, -1], returned_recon4inf), dim=0)
                    imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')
                    imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
                    grid = make_grid(imgs_and_recons, nrow=3, normalize=True, value_range=(0, 1))
                    save_image(grid, 'test.jpg')

                    # compute output
                    logit, image_features, text_features, logit_scale, _ = model(tokens4alignment, None)

                    target = label_all.cuda()

                    output = logit.view(images.shape[0], 4)

                    pred_cls1 = output.argmax(dim=1)
                    correct1 += pred_cls1.eq(target.view_as(pred_cls1)).sum().item()

                    cm = confusion_matrix(target.cpu().numpy(), pred_cls1.cpu().numpy(), labels=np.arange(4))
                    print("Confusion Matrix:")
                    print(cm)

                    if i == 0:
                        all_predicted = pred_cls1
                        all_targets = target
                    else:
                        all_predicted = torch.cat((all_predicted, pred_cls1), 0)
                        all_targets = torch.cat((all_targets, target), 0)

            war = 100. * correct1 / len(val_loader.dataset)

            uar = recall_score(all_targets.cpu().numpy(), all_predicted.cpu().numpy(), average='macro')
            uf1 = f1_score(all_targets.cpu().numpy(), all_predicted.cpu().numpy(), average='macro')

            if correct1 > best_correct:
                best_correct = correct1
                best_all_predicted = all_predicted
                best_all_targets = all_targets

                best_uar = uar
                best_uf1 = uf1

            if correct1 == best_correct:
                if (uar + uf1) > (best_uar + best_uf1):
                    best_uar = uar
                    best_uf1 = uf1
                    best_all_predicted = all_predicted
                    best_all_targets = all_targets

            print("acc: %.3f, best: %.3f" % (war, 100. * best_correct / len(val_loader.dataset)))
            print("uar: %.3f, best: %.3f" % (uar, best_uar))
            print("uf1: %.3f, best: %.3f" % (uf1, best_uf1))

            if best_correct == len(val_loader.dataset):
                break

        best_correct_all += best_correct
        samples_all += len(val_loader.dataset)

        try:
            overall_targets = torch.cat((overall_targets, best_all_targets), 0)
            overall_predicted = torch.cat((overall_predicted, best_all_predicted), 0)
        except NameError:
            overall_targets = best_all_targets
            overall_predicted = best_all_predicted

        current_uar = recall_score(overall_targets.cpu().numpy(), overall_predicted.cpu().numpy(), average='macro')
        current_uf1 = f1_score(overall_targets.cpu().numpy(), overall_predicted.cpu().numpy(), average='macro')

        cm_overall = confusion_matrix(overall_targets.cpu().numpy(), overall_predicted.cpu().numpy(), labels=np.arange(4))
        print(cm_overall)

        print(best_correct_all, samples_all, current_uar, current_uf1)

        del model, optimizer_model, optimizer_laq
        del train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a:[N,D], b:[M,D] -> [N,M]  余弦相似度，要求输入已归一化或内部归一化"""
    # a = F.normalize(a, dim=-1)
    # b = F.normalize(b, dim=-1)
    return a @ b.t()

class AnchorCLIP5(nn.Module):
    """
    4分类，其中 class 3 = others（无文本原型）。
    text_proto: [3, D]  仅包含 0,1,2 3类的文本原型
    训练损失 = 监督CLIP(已知类) + λ_bg*背景边界(others) + λ_ins*实例对比(可选)
    """

    def __init__(self, temperature=0.07, bg_margin=0.2, lambda_bg=0.2):
        super().__init__()
        self.tau = temperature
        self.bg_m = bg_margin
        self.l_bg = lambda_bg

    def forward(
            self,
            img_emb,
            labels,
            text_proto,
            ins_pair=None
    ):
        B, D = img_emb.shape
        assert text_proto.shape == (3, D), "text_proto should be [3, D] (without 'others')"

        v = F.normalize(img_emb, dim=-1)  # [B, D]
        t = text_proto.detach()

        # ---- 划分已知类/others ----
        known_mask = (labels != 3)  # True 表示 0..3
        others_mask = ~known_mask

        # 1) 已知类监督 CLIP（v vs t 的 softmax-CE）
        loss_sup = v.new_zeros(())
        if known_mask.any():
            v_known = v[known_mask]  # [Bk, D]
            y_known = labels[known_mask]  # [Bk] in {0,1,2}
            logits = cosine_sim(v_known, t) / self.tau  # [Bk, 3]
            loss_sup = F.cross_entropy(logits, y_known)  # 仅对 0..2 做监督

        # 2) others 背景边界：others 与任一文本原型的最大相似度 ≤ margin
        loss_bg = v.new_zeros(())
        if others_mask.any():
            v_oth = v[others_mask]  # [Bo, D]
            sim_max = cosine_sim(v_oth, t).max(dim=1).values  # [Bo]
            loss_bg = F.relu(sim_max - self.bg_m).mean()

        total = loss_sup + self.l_bg * loss_bg
        stats = {
            "sup": float(loss_sup.detach().cpu()),
            "bg": float(loss_bg.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        return total, stats


def train(train_loader, laq, model, prompt, criterion, optimizer_laq, optimizer_model, epoch, args, log_txt_path):
    losses_cls = AverageMeter('Loss_cls', ':.4f')
    losses_con = AverageMeter('Loss_con', ':.4f')

    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses_cls, losses_con, top1],
                             prefix="Epoch: [{}]".format(epoch),
                             log_txt_path=log_txt_path)

    # switch to train mode
    model.train()
    laq.train()

    for i, (images, label_all) in enumerate(train_loader):

        images = images.cuda()
        n, _, _, _, _ = images.shape
        target = label_all.cuda()

        loss_rec, num_unique_indices, tokens4alignment, indices, _, _ = laq(
            images,
        )

        class_results, image_features, text_features, logit_scale, sparse_loss = model(tokens4alignment.detach(), None)

        anchorclip = AnchorCLIP5()
        loss_con, _ = anchorclip(image_features, target, prompt)
        loss_cls = criterion(class_results, target)

        loss = loss_cls + 1e-3 * sparse_loss + 0.3 * loss_con
        
        # measure accuracy and record loss
        acc1 = accuracy(class_results, target, topk=(1,))
        losses_cls.update(loss_cls.item(), images.size(0))
        losses_con.update(loss_con.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        optimizer_laq.zero_grad()
        (0.1*loss_rec).backward()
        optimizer_laq.step()

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)
            
    return top1.avg, losses_cls.avg, losses_con.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        # with open(self.log_txt_path, 'a') as f:
        #     f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses_cls = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_losses_con = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]

        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss_cls, train_loss_con, train_acc):
        self.epoch_losses_cls[idx, 0] = train_loss_cls * 50
        self.epoch_losses_con[idx, 0] = train_loss_con * 50

        self.epoch_accuracy[idx, 0] = train_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='r', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses_cls[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses_cls[:, 1]
        plt.plot(x_axis, y_axis, color='r', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
