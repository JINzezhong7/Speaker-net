# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        min_lr,
        max_lr,
        warmup_epoch,
        fix_epoch,
        step_per_epoch
    ):
        self.optimizer = optimizer
        assert min_lr <= max_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_step = warmup_epoch * step_per_epoch
        self.fix_step = fix_epoch * step_per_epoch
        self.current_step = 0.0

    def set_lr(self,):
        new_lr = self.clr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def step(self, step=None):
        if step is not None:
            self.current_step = step
        new_lr = self.set_lr()
        self.current_step += 1
        return new_lr

    def clr(self, step):
        if step < self.warmup_step:
            return self.min_lr + (self.max_lr - self.min_lr) * \
                (step / self.warmup_step)
        elif step >= self.warmup_step and step < self.fix_step:
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                (1 + math.cos(math.pi * (step - self.warmup_step) /
                (self.fix_step - self.warmup_step)))
        else:
            return self.min_lr

class StepScheduler:
    def __init__(
        self,
        optimizer,
        lr,
        step_per_epoch,
        step_epoch_size,
    ):
        self.optimizer = optimizer
        self.lr = lr
        self.step_size = step_epoch_size * step_per_epoch
        self.current_step = 0.0

    def set_lr(self,):
        new_lr = self.clr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def step(self, step=None):
        if step is not None:
            self.current_step = step
        new_lr = self.set_lr()
        self.current_step += 1
        return new_lr

    def clr(self, step):
        ratio = 0.1**(step // self.step_size)
        return self.lr * ratio


class MarginScheduler:
    def __init__(
        self,
        criterion,
        increase_start_epoch,
        fix_epoch,
        step_per_epoch,
        initial_margin,
        final_margin,
        increase_type='exp',
    ):
        assert hasattr(criterion, 'update'), "Loss function not has 'update()' attributes."
        self.criterion = criterion
        self.increase_start_step = increase_start_epoch * step_per_epoch
        self.fix_step = fix_epoch * step_per_epoch
        self.initial_margin = initial_margin
        self.final_margin = final_margin
        self.increase_type = increase_type
        self.margin = initial_margin

        self.current_step = 0
        self.increase_step = self.fix_step - self.increase_start_step

        self.init_margin()

    def init_margin(self):
        self.criterion.update(margin=self.initial_margin)

    def step(self, current_step=None):
        if current_step is not None:
            self.current_step = current_step

        self.margin = self.iter_margin()
        self.criterion.update(margin=self.margin)
        self.current_step += 1

    def iter_margin(self):
        if self.current_step < self.increase_start_step:
            return self.initial_margin

        if self.current_step >= self.fix_step:
            return self.final_margin

        a = 1.0
        b = 1e-3

        current_step = self.current_step - self.increase_start_step
        if self.increase_type == 'exp':
             # exponentially increase the margin
            ratio = 1.0 - math.exp(
                (current_step / self.increase_step) *
                math.log(b / (a + 1e-6))) * a
        else:
            # linearly increase the margin
            ratio = 1.0 * current_step / self.increase_step
        return self.initial_margin + (self.final_margin -
                                      self.initial_margin) * ratio

    def get_margin(self):
        return self.margin

class ExponetialDecrease():
    def __init__(
        self,
        optimizer,
        min_lr,
        max_lr,
        num_epoch,
        step_per_epoch,
        warmup_epoch,
        scale_ratio=1.0,
        warm_from_zero=False
    ):
        self.optimizer = optimizer
        assert min_lr <= max_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_step = warmup_epoch * step_per_epoch
        self.max_step = num_epoch * step_per_epoch
        self.current_step = 0.0
        self.warm_from_zero = warm_from_zero
        self.scale_ratio = scale_ratio

    def get_multi_process_coeff(self):
        lr_coeff = 1.0 * self.scale_ratio
        if self.current_step < self.warmup_step:
            if self.warm_from_zero:
                lr_coeff = self.scale_ratio * self.current_step / self.warmup_step
            elif self.scale_ratio >1:
                lr_coeff = (self.scale_ratio -
                            1) * self.current_step / self.warmup_step + 1.0
        return lr_coeff

    def get_current_lr(self):
        lr_coeff = self.get_multi_process_coeff()
        current_lr = lr_coeff * self.max_lr * math.exp(
            (self.current_step / self.max_step) *
            math.log(self.min_lr / self.max_lr)
        )

        return current_lr

    def set_lr(self,):
        new_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def step(self, current_step=None):
        if current_step is not None:
            self.current_step = current_step
        new_lr = self.set_lr()
        self.current_step += 1

        return new_lr
    