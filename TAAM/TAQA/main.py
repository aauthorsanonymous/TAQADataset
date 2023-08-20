import os
import sys

sys.path.append('../')

import torch
import torch.nn as nn

from utils import *
from opts import *
from scipy import stats
from tqdm import tqdm
from dataset import VideoDataset
from models.i3d import InceptionI3d
from models.evaluator import Evaluator
from config import get_parser


def get_models(args):
    """
    Get the i3d backbone and the evaluator with parameters moved to GPU.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    i3d = InceptionI3d().cuda()
    i3d.load_state_dict(torch.load(i3d_pretrained_path))


    evaluator = Evaluator(output_dim=output_dim['TAAM'], num_judges=num_judges).cuda()

    if len(args.gpu.split(',')) > 1:
        i3d = nn.DataParallel(i3d)
        evaluator = nn.DataParallel(evaluator)
    return i3d, evaluator


def compute_score(probs, data):
    # calculate expectation & denormalize & sort
    judge_scores_pred = torch.stack([prob.argmax(dim=-1) * judge_max / (output_dim['TAAM']-1)
                                         for prob in probs], dim=1).sort()[0]  # N, 7


    pred = torch.sum(judge_scores_pred[:, 0:5], dim=1) * data['difficulty'].cuda()
    return pred


def compute_loss(criterion, probs, data):
    loss = sum([criterion(torch.log(probs[i]), data['soft_judge_scores'][:, i].cuda()) for i in range(num_judges)])
    return loss


def get_dataloaders(args):
    dataloaders = {}

    dataloaders['train'] = torch.utils.data.DataLoader(VideoDataset('train', args),
                                                       batch_size=args.train_batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       pin_memory=True,
                                                       worker_init_fn=worker_init_fn)

    dataloaders['test'] = torch.utils.data.DataLoader(VideoDataset('test', args),
                                                      batch_size=args.test_batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      worker_init_fn=worker_init_fn)
    return dataloaders


def main(dataloaders, i3d, evaluator, base_logger, args):
    # print configuration
    print('=' * 40)
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 40)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([*i3d.parameters()] + [*evaluator.parameters()],
                                 lr=args.lr, weight_decay=args.weight_decay)

    epoch_best = 0
    rho_best = 0
    for epoch in range(args.num_epochs):
        log_and_print(base_logger, f'Epoch: {epoch}  Current Best: {rho_best} at epoch {epoch_best}')

        for split in ['train', 'test']:
            true_scores = []
            pred_scores = []

            if split == 'train':
                i3d.train()
                evaluator.train()
                torch.set_grad_enabled(True)
            else:
                i3d.eval()
                evaluator.eval()
                torch.set_grad_enabled(False)

            for data in tqdm(dataloaders[split]):
                true_scores.extend(data['final_score'].numpy())
                videos = data['video'].cuda()
                videos.transpose_(1, 2)  # N, C, T, H, W

                batch_size, C, frames, H, W = videos.shape
                clip_feats = torch.empty(batch_size, 10, feature_dim).cuda()
                for i in range(9):
                    clip_feats[:, i] = i3d(videos[:, :, 10 * i:10 * i + 16, :, :]).squeeze(2)
                clip_feats[:, 9] = i3d(videos[:, :, -16:, :, :]).squeeze(2)

                probs = evaluator(clip_feats.mean(1))
                preds = compute_score(probs, data)
                pred_scores.extend([i.item() for i in preds])

                if split == 'train':
                    loss = compute_loss(criterion, probs, data)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            rho, p = stats.spearmanr(pred_scores, true_scores)

            log_and_print(base_logger, f'{split} correlation: {rho}')

        if rho > rho_best:
            rho_best = rho
            epoch_best = epoch
            log_and_print(base_logger, '-----New best found!-----')
            if args.save:
                torch.save({'epoch': epoch,
                            'i3d': i3d.state_dict(),
                            'evaluator': evaluator.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'rho_best': rho_best}, f'work_dir/TAAM.pt')


if __name__ == '__main__':

    args = get_parser().parse_args()

    if not os.path.exists('./work_dir'):
        os.mkdir('./work_dir')
    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')

    init_seed(args)

    base_logger = get_logger(f'work_dir/TAAM.log', args.log_info)
    i3d, evaluator = get_models(args)
    dataloaders = get_dataloaders(args)

    main(dataloaders, i3d, evaluator, base_logger, args)
