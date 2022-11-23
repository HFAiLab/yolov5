import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler

import hfai
from hfai.nn.parallel import DistributedDataParallel
# from torch.nn.parallel import DistributedDataParallel

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from val import process_batch  # for end-of-epoch mAP
from models.yolo import Model
from utils.dataloaders_hfai import create_dataloader
from utils.general import (LOGGER, Profile, check_amp, non_max_suppression, check_img_size, init_seeds, one_cycle, scale_boxes, xywh2xyxy)
from utils.loss import ComputeLoss
from utils.metrics import fitness, ConfusionMatrix, ap_per_class
from utils.torch_utils import (ModelEMA, smart_optimizer)


def load_model(model, optimizer=None, lr_scheduler=None, ema=None, save_path=None):

    start_epoch, start_step = 0, 0
    best_fitness = 0.
    if save_path.exists():
        ckpt = torch.load(save_path, map_location="cpu")
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        start_epoch = ckpt["epoch"]
        start_step = ckpt["step"]
        best_fitness = ckpt["best_fitness"]
        if ema:
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
            ema.updates = ckpt['updates']
    else:
        model.half().float()  # pre-reduce anchor precision

    return start_epoch, start_step, best_fitness


def save_model(model, epoch=0, step=0, opt=None, optimizer=None, lr_scheduler=None, ema=None, best_fitness=0., save_path=None):
    ckpt = {
        'epoch': epoch,
        'step': step,
        'best_fitness': best_fitness,
        'model': model.module.state_dict(),
        'ema': ema.ema.state_dict(),
        'updates': ema.updates,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'opt': vars(opt)}
    torch.save(ckpt, save_path)


def valid(half=True, model=None, dataloader=None, single_cls=False, save_dir=Path(''), plots=True, compute_loss=None):

    # Configure
    model.eval()
    nc = 128
    iouv = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)

    tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    dt = Profile(), Profile(), Profile()  # profiling times
    loss = torch.zeros(3).cuda()
    jdict, stats, ap, ap_class = [], [], [], []

    for batch_i, (im, targets, shapes) in enumerate(dataloader):

        with dt[0]:
            im = im.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        with dt[1]:
            preds, train_out = model(im) if compute_loss else (model(im, augment=False), None)

        # Loss
        if compute_loss:
            loss += compute_loss(train_out, targets)[1]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
        lb = []  # for autolabelling
        with dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres=0.001,
                                        iou_thres=0.6,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=300)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            shape = shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool).cuda()  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0)).cuda(), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, save_dir=save_dir)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Return results
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps


def train(rank, local_rank, hyp, opt):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    if local_rank in {-1, 0}:
        print('hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    init_seeds(opt.seed + 1 + rank, deterministic=True)
    nc = 128

    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors'))
    model = hfai.nn.to_hfai(model)
    amp = check_amp(model)  # check AMP

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            if local_rank == 0:
                print(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # DDP mode
    if rank != -1:
        model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])

    # EMA
    ema = ModelEMA(model) if rank in {-1, 0} else None

    # Resume
    best_fitness, start_epoch, start_step = load_model(model, optimizer, scheduler, ema, last)

    # SyncBatchNorm
    if opt.sync_bn and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if local_rank == 0:
            print('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader('train',
                                              imgsz,
                                              batch_size,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None,
                                              rect=opt.rect,
                                              rank=local_rank,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              shuffle=True)

    if rank in {-1, 0} and local_rank in {-1, 0}:
        val_loader = create_dataloader('val',
                                       imgsz,
                                       batch_size,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5)[0]

    # Start training
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    compute_loss = ComputeLoss(model, hyp)  # init loss class
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(3).cuda()  # mean losses
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)

        optimizer.zero_grad()
        trecords = {'fwd': 0., 'opt': 0., 'bwd': 0.}
        for step, (imgs, targets, _) in enumerate(train_loader):  # batch -------------------------------------------------------------
            if step < start_step:
                continue

            imgs = imgs.float().cuda(non_blocking=True) / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            t0 = time.time()
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.cuda(non_blocking=True))  # loss scaled by batch_size
                if opt.quad:
                    loss *= 4.

            # Backward
            t1 = time.time()
            scaler.scale(loss).backward()

            t2 = time.time()
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
            t3 = time.time()

            trecords['fwd'] += t1 - t0
            trecords['bwd'] += t2 - t1
            trecords['opt'] += t3 - t2

            # Log
            if local_rank in {-1, 0}:
                mloss = (mloss * step + loss_items) / (step + 1)  # update mean losses

            if rank in {-1, 0} and local_rank in {-1, 0} and hfai.receive_suspend_command():
                save_model(model, epoch, step+1, opt, optimizer, scheduler, ema, best_fitness, last)
                time.sleep(5)
                hfai.go_suspend()

            torch.cuda.empty_cache()

        start_step = 0
        scheduler.step()

        if rank in {-1, 0} and local_rank in {-1, 0}:
            print(f"Epoch: {epoch} train | box_loss: {mloss[0]:.5f}, obj_loss: {mloss[1]:.5f}, cls_loss: {mloss[2]:.5f} ")
            print(f"Epoch: {epoch} time | forward: {trecords['fwd']:.2f}s, optimize: {trecords['opt']:.2f}s, backward: {trecords['bwd']:.2f}s")

            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            results, maps = valid(half=amp,
                                  model=ema.ema,
                                  single_cls=single_cls,
                                  dataloader=val_loader,
                                  save_dir=save_dir,
                                  plots=False,
                                  compute_loss=compute_loss)
            print(f"Epoch: {epoch} valid | P: {results[0]:.5f}, R: {results[1]:.5f}, mAP@.5: {results[2]:.5f}")

            # Save model
            save_model(model, epoch+1, 0, opt, optimizer, scheduler, ema, best_fitness, last)
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
                save_model(model, epoch+1, 0, opt, optimizer, scheduler, ema, best_fitness, best)

        # end epoch ----------------------------------------------------------------------------------------------------

    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(local_rank, opt):
    # Checks
    if local_rank in {-1, 0}:
        print('train: ' + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    opt.save_dir = Path(str(opt.project)) / opt.name
    opt.hyp = str(opt.hyp)
    rank = -1

    # DDP mode
    if local_rank != -1:
        # init dist
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "54247")
        hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
        rank = int(os.environ.get("RANK", "0"))  # node id
        gpus = torch.cuda.device_count()  # gpus per node

        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank)
        torch.cuda.set_device(local_rank)

    # Train
    train(rank, local_rank, opt.hyp, opt)


if __name__ == "__main__":
    opt = parse_opt()
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(opt,), nprocs=ngpus, bind_numa=True)
    # main(local_rank=-1, opt=opt)
