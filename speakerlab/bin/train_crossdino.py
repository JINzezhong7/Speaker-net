# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import json
import math
import numpy
import argparse
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import speakerlab.utils.utils as utils
import speakerlab.utils.utils_rdino as utils_rdino
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build

parser = argparse.ArgumentParser(description='Regularized DINO Framework Training')
parser.add_argument('--config', default='', type=str, help='Config file')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')

def main():
    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, overrides, True)

    # DDP
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(args.gpu[rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl")

    utils_rdino.setup_for_distributed(rank == 0)
    utils.set_seed(args.seed)

    model_save_path = config.exp_dir + "/models"
    log_save_path = config.exp_dir

    if rank == 0:
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(log_save_path, exist_ok=True)

    # Initialise data loader
    train_dataset = build('dataset', config)
    sampler = torch.utils.data.DistributedSampler(train_dataset)
    # build dataloader
    config.dataloader['args']['sampler'] = sampler
    train_loader = build('dataloader', config)

    print(f"Data loaded: there are {len(train_loader)} iterations.")

    # Load models
    student_ecapa = build('student_ecapa', config)
    teacher_ecapa = build('teacher_ecapa', config)
    student_resnet= build('student_resnet',config)
    teacher_resnet= build('teacher_resnet',config)

    # synchronize batch norms
    student_ecapa = nn.SyncBatchNorm.convert_sync_batchnorm(student_ecapa)
    teacher_ecapa = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_ecapa)
    student_resnet= nn.SyncBatchNorm.convert_sync_batchnorm(student_resnet)
    teacher_resnet= nn.SyncBatchNorm.convert_sync_batchnorm(teacher_resnet)
    student_ecapa.cuda()
    teacher_ecapa.cuda()
    student_resnet.cuda()
    teacher_resnet.cuda()

    # DDP wrapper
    student_ecapa = nn.parallel.DistributedDataParallel(student_ecapa)
    teacher_ecapa = nn.parallel.DistributedDataParallel(teacher_ecapa)
    student_resnet = nn.parallel.DistributedDataParallel(student_resnet)
    teacher_resnet = nn.parallel.DistributedDataParallel(teacher_resnet)

    # teacher and student start with the same weights
    teacher_ecapa.load_state_dict(student_ecapa.state_dict())
    teacher_resnet.load_state_dict(student_resnet.state_dict())

    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher_ecapa.parameters():
        p.requires_grad = False
    
    for p in teacher_resnet.parameters():
        p.requires_grad = False
    
    print("Student and teacher are using the same network architecture but different parameters.")

    # Prepare dino loss
    self_loss_ecapa = build('self_loss_ecapa',config).cuda()
    self_loss_resnet = build('self_loss_resnet',config).cuda()
    cross_loss_ecapa = build('cross_loss_ecapa',config).cuda()
    cross_loss_resnet = build('cross_loss_resnet',config).cuda()
    # Prepare optimizer
    params_groups_ecapa = utils_rdino.get_params_groups(student_ecapa)
    if config.optimizer =="sgd":
        optimizer_ecapa = torch.optim.SGD(params_groups_ecapa, lr=0, momentum=0.9)
    elif config.optimizer =="adamw":
        optimizer_ecapa = torch.optim.AdamW(params_groups_ecapa)

    param_groups_resnet = utils_rdino.get_params_groups(student_resnet)
    optimizer_resnet = torch.optim.SGD(param_groups_resnet,lr=0,momentum=0.9)
    # Prepare learning rate schedule and weight decay scheduler
    lr_schedule_ecapa = utils_rdino.cosine_scheduler(
        config.lr_ecapa * (config.batch_size_per_gpu * utils_rdino.get_world_size()) / 256.,
        config.min_lr,
        config.epochs,
        len(train_loader),
        warmup_epochs=config.warmup_epochs,
    )

    wd_schedule_ecapa = utils_rdino.cosine_scheduler(
        config.weight_decay,
        config.weight_decay_end,
        config.epochs,
        len(train_loader),
    )

    lr_schedule_resnet = utils_rdino.cosine_scheduler(
        config.lr_resnet * (config.batch_size_per_gpu * utils_rdino.get_world_size()) / 256.,
        config.min_lr,
        config.epochs,
        len(train_loader),
        warmup_epochs=config.warmup_epochs,
    )

    wd_schedule_resnet = utils_rdino.cosine_scheduler(
        config.weight_decay,
        config.weight_decay_end,
        config.epochs,
        len(train_loader),
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils_rdino.cosine_scheduler(
        config.momentum_teacher,
        1,
        config.epochs,
        len(train_loader)
    )

    # Resume training optionally
    to_restore = {"epoch": 0}
    utils_rdino.restart_from_checkpoint(
        os.path.join(model_save_path, "checkpoint.pth"),
        run_variables=to_restore,
        student_ecapa=student_ecapa,
        teacher_ecapa=teacher_ecapa,
        student_resnet=student_resnet,
        teacher_resnet=teacher_resnet,
        optimizer_ecapa=optimizer_ecapa,
        optimizer_resnet=optimizer_resnet,
        self_loss_ecapa=self_loss_ecapa,
        self_loss_resnet=self_loss_resnet,
        cross_loss_ecapa=cross_loss_ecapa,
        cross_loss_resnet=cross_loss_resnet,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting CROSS-DISSTILLATION DINO training !")

    for epoch in range(start_epoch, config.epochs):
        train_loader.sampler.set_epoch(epoch)
        # Training one epoch
        train_stats = train_one_epoch(student_ecapa, teacher_ecapa, student_resnet, teacher_resnet, \
        self_loss_ecapa, self_loss_resnet, cross_loss_ecapa, cross_loss_resnet, \
        train_loader, optimizer_ecapa, optimizer_resnet, lr_schedule_ecapa, \
        lr_schedule_resnet, wd_schedule_ecapa, wd_schedule_resnet, momentum_schedule, epoch, config)

        # Write logs
        save_dict = {
            'student_ecapa': student_ecapa.state_dict(),
            'teacher_ecapa': teacher_ecapa.state_dict(),
            'student_resnet': student_resnet.state_dict(),
            'teacher_resnet': teacher_resnet.state_dict(),
            'optimizer_ecapa': optimizer_ecapa.state_dict(),
            'optimizer_resnet': optimizer_resnet.state_dict(),
            'epoch': epoch + 1,
            'self_loss_ecapa': self_loss_ecapa.state_dict(),
            'self_loss_resnet': self_loss_resnet.state_dict(),
            'cross_loss_ecapa': cross_loss_ecapa.state_dict(),
            'cross_loss_resnet': cross_loss_resnet.state_dict(),
        }

        utils_rdino.save_on_master(save_dict, os.path.join(model_save_path, 'checkpoint.pth'))
        if config.saveckp_freq and epoch % config.saveckp_freq == 0:
            utils_rdino.save_on_master(save_dict, os.path.join(model_save_path, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if utils_rdino.is_main_process():
            with  open(log_save_path + "/log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")


def train_one_epoch(student_ecapa, teacher_ecapa, student_resnet, teacher_resnet, \
        self_loss_ecapa, self_loss_resnet, cross_loss_ecapa, cross_loss_resnet, \
        train_loader, optimizer_ecapa, optimizer_resnet, lr_schedule_ecapa, \
        lr_schedule_resnet, wd_schedule_ecapa, wd_schedule_resnet, momentum_schedule, epoch, config):

    teacher_ecapa.train()
    student_ecapa.train()
    teacher_resnet.train()
    student_resnet.train()

    metric_logger = utils_rdino.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch+1, config.epochs)
    for it, data in enumerate(
        metric_logger.log_every(
            train_loader, print_freq=100, header=header)):
        with torch.autograd.set_detect_anomaly(True):
            # update weight decay and learning rate according to their schedule
            it = len(train_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optimizer_ecapa.param_groups):
                param_group["lr"] = lr_schedule_ecapa[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule_ecapa[it]
            
            for i, param_group in enumerate(optimizer_resnet.param_groups):
                param_group["lr"] = lr_schedule_resnet[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule_resnet[it]

            # Data.shape is [batch_size, glb_num + int(local_num / 2), fbank_dim, frames]
            # Local_frames is half of the global_frames (max_frames)
            batch_size, _, fbank_dim, frame_dim = data.shape
            data = data.transpose(0,1).cuda()
            global_data = data[0: config.glb_num, :, :, :]
            global_data = global_data.reshape(-1, fbank_dim, frame_dim)
            teacher_output_ecapa = teacher_ecapa(global_data)
            student_global_output_ecapa = student_ecapa(global_data)
            teacher_output_resnet = teacher_resnet(global_data)
            student_global_output_resnet = student_resnet(global_data)
            local_data1 = data[config.glb_num, :, :, :]
            local_data2 = data[config.glb_num + 1, :, :, :]
            local_frames = config.local_frames
            local_data1_1 = local_data1[:, :, 0: local_frames]
            local_data1_2 = local_data1[:, :, local_frames: ]
            local_data2_1 = local_data2[:, :, 0: local_frames]
            local_data2_2 = local_data2[:, :, local_frames:]
            local_data = torch.cat((local_data1_1,local_data1_2,local_data2_1,local_data2_2), axis=0)
            # local_data = torch.cat((local_data1_1,local_data1_2),axis=0)

            student_local_output_ecapa = student_ecapa(local_data)
            student_local_output_resnet = student_resnet(local_data)
            student_output_ecapa = torch.cat((student_global_output_ecapa, student_local_output_ecapa), axis=0)
            student_output_resnet = torch.cat((student_global_output_resnet,student_local_output_resnet), axis=0)
            # locals = torch.cat(torch.unbind(locals,dim=1),dim=0).cuda()
            # globals = torch.cat(torch.unbind(globals,dim=1),dim=0).cuda()

            # # student ecapa
            # student_ecapa_g = student_ecapa(globals)
            # student_ecapa_l = student_ecapa(locals)
            # student_output_ecapa = torch.cat((student_ecapa_g,student_ecapa_l),axis=0)

            # # student resnet
            # student_resnet_g = student_resnet(globals)
            # student_resnet_l = student_resnet(locals)
            # student_output_resnet = torch.cat((student_resnet_g, student_resnet_l),axis=0)

            # # teacher ecapa
            # with torch.no_grad():
            #     teacher_output_ecapa = teacher_ecapa(globals)

            # # teacher resnet
            # with torch.no_grad():
            #     teacher_output_resnet = teacher_resnet(globals)

            # loss
            self_loss_e = self_loss_ecapa(student_output_ecapa, teacher_output_ecapa, epoch)
            cross_loss_e = cross_loss_ecapa(student_output_ecapa, teacher_output_resnet, epoch)
            loss_ecapa_total = self_loss_e + config.lamda_ecapa * cross_loss_e 

            self_loss_r = self_loss_resnet(student_output_resnet, teacher_output_resnet, epoch)
            cross_loss_r = cross_loss_resnet(student_output_resnet, teacher_output_ecapa, epoch)
            loss_resnet_total = self_loss_r + config.lamda_resnet * cross_loss_r
            if not math.isfinite(loss_ecapa_total.item()):
                raise Exception("Loss is {}, stopping training".format(loss_ecapa_total.item()), force=True)
            if not math.isfinite(loss_resnet_total.item()):
                raise Exception("Loss is {}, stopping training".format(loss_resnet_total.item()),force=True)

            # ECAPA student network update
            optimizer_ecapa.zero_grad()
            param_norms = None
            loss_ecapa_total.backward()
            if config.clip_grad_ecapa:
                param_norms = utils_rdino.clip_gradients(student_ecapa, config.clip_grad_ecapa)
            utils_rdino.cancel_gradients_last_layer(epoch, student_ecapa, config.freeze_last_layer)
            optimizer_ecapa.step()

            # EMA update for the ECAPA teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student_ecapa.parameters(), teacher_ecapa.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            
            # Resnet student network update
            optimizer_resnet.zero_grad()
            param_norms = None
            loss_resnet_total.backward()
            if config.clip_grad_resnet:
                param_norms = utils_rdino.clip_gradients(student_resnet, config.clip_grad_resnet)
            utils_rdino.cancel_gradients_last_layer(epoch, student_resnet, config.freeze_last_layer)
            optimizer_resnet.step()

            #EMA update for the Resnet teacher
            with torch.no_grad():
                m = momentum_schedule[it]
                for param_q, param_k in zip(student_resnet.parameters(), teacher_resnet.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss_ecapa=loss_ecapa_total.item())
            metric_logger.update(loss_resnet=loss_resnet_total.item())
            metric_logger.update(lr_ecapa=optimizer_ecapa.param_groups[0]["lr"])
            metric_logger.update(lr_resnet=optimizer_resnet.param_groups[0]["lr"])
            # metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()