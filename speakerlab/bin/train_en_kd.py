# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from speakerlab.models.en_kd.speaker_model import get_speaker_model
from speakerlab.utils.utils import set_seed, get_logger, AverageMeters, ProgressMeter, accuracy, load_params
from speakerlab.utils.config import build_config
from speakerlab.utils.builder import build
from speakerlab.utils.epoch import EpochCounter, EpochLogger
from thop import profile

parser = argparse.ArgumentParser(description='Speaker Network Training')
parser.add_argument('--config', default='', type=str, help='Config file for training')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')
parser.add_argument('--seed', default=1234, type=int, help='Random seed for training.')
parser.add_argument('--gpu', nargs='+', help='GPU id to use.')

def recover_embedding_model(path,teacher_config,embedding_model,classifier):
    teacher_config.checkpointer['args']['checkpoints_dir'] = os.path.join(path, 'models')
    teacher_config.checkpointer['args']['recoverables'] = {'embedding_model':embedding_model,'classifier':classifier}
    checkpointer = build('checkpointer', teacher_config)
    # checkpointer.recover_if_possible(epoch=config.num_epoch, device=device)
    checkpointer.recover_if_possible(epoch=60, device="cuda")

def main():
    args, overrides = parser.parse_known_args(sys.argv[1:])
    config = build_config(args.config, overrides, True)
    # set DDP
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(args.gpu[rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    # set random seed
    set_seed(args.seed)

    os.makedirs(config.exp_dir, exist_ok=True)
    logger = get_logger('%s/train.log' % config.exp_dir)
    logger.info(f"Use GPU: {gpu} for training.")

    # dataset
    train_dataset = build('dataset', config)
    # dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    config.dataloader['args']['sampler'] = train_sampler
    config.dataloader['args']['batch_size'] = int(config.batch_size / world_size)
    train_dataloader = build('dataloader', config)
 
    # build models
    student_embedding = build('embedding_student',config)
    teacher_model = get_speaker_model(config.teacher_model)(**config.teacher_model_args)
    teacher_model.eval()
    if hasattr(config, 'speed_pertub') and config.speed_pertub:
        config.num_classes = len(config.label_encoder) * 3
    else:
        config.num_classes = len(config.label_encoder)

    student_classifier = build('student_classifier', config)
    student_model = nn.Sequential(student_embedding, student_classifier)
    
    # Compute the param & flops of student model
    test_data = torch.randn(1,298,80)
    flops, params = profile(student_model, inputs=(test_data,))

    student_model.cuda()
    teacher_model.cuda()

    # there is no backprogation through the teacher, so no need for gradients
    for p in teacher_model.parameters():
        p.requires_grad = False

    #DDP wrapper
    student_model = torch.nn.parallel.DistributedDataParallel(student_model)
    teacher_model = torch.nn.parallel.DataParallel(teacher_model,device_ids=[gpu])


    # optimizer
    config.optimizer['args']['params'] = student_model.parameters()
    optimizer = build('optimizer', config)

    # loss function
    aam = build('AAM', config)
    aux_criterion = build('DKD_LOSS', config)
    cross_entropy = torch.nn.CrossEntropyLoss()
    # scheduler
    config.lr_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    config.lr_scheduler['args']['scale_ratio'] = 1.0 * world_size * config.batch_size /64
    lr_scheduler = build('lr_scheduler', config)
    config.margin_scheduler['args']['step_per_epoch'] = len(train_dataloader)
    margin_scheduler = build('margin_scheduler', config)
 
    # others
    epoch_counter = build('epoch_counter', config)
    checkpointer = build('checkpointer', config)

    epoch_logger = EpochLogger(save_file=os.path.join(config.exp_dir, 'train_epoch.log'))

    # resume from a checkpoint
    if args.resume:
        checkpointer.recover_if_possible(device='cuda')

    cudnn.benchmark = True

    for epoch in epoch_counter:
        train_sampler.set_epoch(epoch)

        # train one epoch
        train_stats = train(
            train_dataloader,
            student_model,
            teacher_model,
            aam,
            aux_criterion,
            cross_entropy,
            optimizer,
            epoch,
            lr_scheduler,
            margin_scheduler,
            logger,
            config,
            rank,
        )

        if rank == 0:
            # log
            epoch_logger.log_stats(
                stats_meta={"epoch": epoch},
                stats=train_stats,
            )
            # save checkpoint
            if epoch % config.save_epoch_freq == 0:
                checkpointer.save_checkpoint(epoch=epoch)

        dist.barrier()

def train(train_loader, student_model,teacher_model, aam, aux_criterion,cross_entropy, optimizer, epoch, lr_scheduler, margin_scheduler, logger, config, rank):
    train_stats = AverageMeters()
    train_stats.add('Time', ':6.3f')
    train_stats.add('Data', ':6.3f')
    train_stats.add('Total_Loss', ':.4e')
    train_stats.add('main_loss', ':.4e')
    train_stats.add('aux_loss', ':.4e')
    train_stats.add('Acc@1', ':6.2f')
    train_stats.add('Lr', ':.3e')
    train_stats.add('Margin', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        train_stats,
        prefix="Epoch: [{}]".format(epoch)
    )

    #train mode
    student_model.train()

    end = time.time()
    for i, (x,feat, y) in enumerate(train_loader):
        # data loading time
        train_stats.update('Data', time.time() - end)

        # update
        iter_num = (epoch-1)*len(train_loader) + i
        lr_scheduler.step(iter_num)
        margin_scheduler.step(iter_num)
        x = x.cuda(non_blocking=True)
        feat = feat.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        teacher_model.eval()
        # compute output
        student_output = student_model(feat)
        student_logits = aam(student_output,y)
        with torch.no_grad():
            teacher_logits = teacher_model(x,y)

        # calculate loss
        main_loss = cross_entropy(student_logits,y) ## AAMSoftmax loss
        aux_loss = aux_criterion(student_logits, teacher_logits, y, epoch)
        loss = main_loss + aux_loss
        acc1 = accuracy(student_output, y)
        
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # recording
        train_stats.update('Total_Loss', loss.item(), x.size(0))
        train_stats.update('main_loss', main_loss.item(), x.size(0))
        train_stats.update('aux_loss', aux_loss.item(), x.size(0))
        train_stats.update('Acc@1', acc1.item(), x.size(0))
        train_stats.update('Lr', optimizer.param_groups[0]["lr"])
        train_stats.update('Margin', margin_scheduler.get_margin())
        train_stats.update('Time', time.time() - end)

        if rank == 0 and i % config.log_batch_freq == 0:
            logger.info(progress.display(i))

        end = time.time()

    key_stats={
        'Avg_loss': train_stats.avg('Total_Loss'),
        'Avg_acc': train_stats.avg('Acc@1'),
        'Lr_value': train_stats.val('Lr')
    }
    return key_stats

if __name__ == '__main__':
    main()
