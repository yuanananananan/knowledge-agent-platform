import argparse
import glob
import logging
import os
import sys
import time
import json

import NAS.genotypes
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import NAS.utils as utils
from NAS.genotypes import Genotype
from .model import NetworkCIFAR as Network
from torch.autograd import Variable

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(1, PROJECT_ROOT)
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# try:
#     from ..dataset.mon import get_MON_dataset
#     from ..dataset.rd import get_rd_dataset
#     from ..dataset.sar import get_sar_dataset_full
from dataset.mon import get_MON_dataset
from dataset.rd import get_rd_dataset
from dataset.sar import get_sar_dataset_full

from utils.utils import load_config, get_dataloader

# MODALITY_CONFIG = {
#     'mon': {
#         'loader': get_MON_dataset,
#         'default_root': 'data/2026split/MON',
#     },
#     'sar': {
#         'loader': get_sar_dataset_full,
#         'default_root': '../data/FUSAR_Ship_3',
#     },
#     'rd': {
#         'loader': get_rd_dataset,
#         'default_root': '../data/RD',
#     },
# }
MODALITY_CONFIG = {'mon', 'sar', 'rd'}

NUM_CLASSES = 4


def build_parser():
    parser = argparse.ArgumentParser('cifar')
    parser.add_argument('--modality', type=str, default='mon', choices=tuple(MODALITY_CONFIG), help='training modality')
    parser.add_argument('--data', type=str, default=None, help='override dataset root directory')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='gpu device id (kept for compatibility; DP uses all visible gpus)')
    parser.add_argument('--epochs', type=int, default=200, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=10, help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--genotype_json_path',type=str,default=None,help='path of searched NAS genotype json')
    return parser


def resolve_dataset_root(args):
    modality = args.modality.lower()
    default_root = MODALITY_CONFIG[modality]['default_root']

    return args.data or default_root


def get_dataset_loader(modality):
    return MODALITY_CONFIG[modality.lower()]['loader']


def setup_logging(save_dir):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


def get_model():
    parser = build_parser()
    args = parser.parse_args(None)
    # args.auxiliary = False  # NAS不使用辅助分类器
    # args.auxiliary = False
    modality = args.modality.lower()
    if args.genotype_json_path:
        genotype = load_genotype_from_json(args.genotype_json_path)
    else:
        genotype_set = NAS.genotypes.DARTS_set
        genotype = genotype_set[modality]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype).to(device)
    net.drop_path_prob = args.drop_path_prob
    return net

def load_genotype_from_json(json_path):
    """
    Restore the NAS Genotype from the genotype.json file sent from the app backend.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Genotype(
        normal=[tuple(x) for x in data["normal"]],
        normal_concat=data["normal_concat"],
        reduce=[tuple(x) for x in data["reduce"]],
        reduce_concat=data["reduce_concat"],
    )

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    modality = args.modality.lower()

    args.save = 'model_ckp/eval-{}-{}{}'.format(args.save, modality, time.strftime('%m%d-%H%M%S'))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    setup_logging(args.save)

    # if not torch.cuda.is_available():
    #     logging.info('no gpu device available')
    #     sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    # data_root = resolve_dataset_root(args)
    # dataset_loader = get_dataset_loader(modality)

    # logging.info('training modality = %s', modality)
    # logging.info('dataset root = %s', data_root)
    logging.info('gpu visible devices = %s', os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL'))
    logging.info('args = %s', args)

    # genotype = eval('NAS.genotypes.%s' % args.arch)
    if args.genotype_json_path:
        genotype = load_genotype_from_json(args.genotype_json_path)
        logging.info("Loaded searched genotype from: %s", args.genotype_json_path)
    else:
        genotype_set = NAS.genotypes.DARTS_set
        genotype = genotype_set[modality]
        logging.info("Loaded default genotype for modality: %s", modality)

    net = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype).cuda()
    model = nn.DataParallel(net).cuda()

    logging.info('param size = %fMB', utils.count_parameters_in_MB(net))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # train_data, valid_data = dataset_loader(data_root)
    cfg = load_config(f'configs/{modality}_config.yaml')
    train_queue, valid_queue = get_dataloader(cfg)

    # train_queue = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=0,
    #     drop_last=True,
    # )

    # valid_queue = torch.utils.data.DataLoader(
    #     valid_data,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=0,
    #     drop_last=True,
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    best_acc = 0.0
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        net.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, args)
        valid_acc, valid_obj = infer(valid_queue, model, criterion, args)
        logging.info('train_acc %f  valid_acc %f', train_acc, valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            logging.info('!!! New Best Validation Accuracy: %f !!!', best_acc)
            utils.save(net, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top2 = utils.AvgrageMeter()
    model.train()

    for step, (input, target, _) in enumerate(train_queue):
        input = Variable(input).cuda(non_blocking=True)
        target = Variable(target).cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(input)

        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)

        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top2.update(prec2.data.item(), n)

        # if step % args.report_freq == 0:
        #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top2.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top2 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target, _) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input).cuda(non_blocking=True)
            target = Variable(target).cuda(non_blocking=True)

        logits, _ = model(input)
        loss = criterion(logits, target)

        prec1, prec2 = utils.accuracy(logits, target, topk=(1, 2))
        n = input.size(0)

        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top2.update(prec2.data.item(), n)

        # if step % args.report_freq == 0:
        #     logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top2.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
