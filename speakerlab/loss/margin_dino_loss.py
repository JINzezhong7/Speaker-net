import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import math
import speakerlab.utils.utils_rdino as utils_rdino

class Margin_DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        ntcrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        threshold=0.6,
        student_temp=0.1,
        center_momentum=0.9
    ):
        super().__init__()

        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.ntcrops= ntcrops
        self.threshold = threshold
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                                                     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

    def custom_function(self,input_matrix, threshold):
        output_matrix = torch.where(input_matrix >threshold, torch.tensor(0).cuda(), torch.tensor(1).cuda())
        return output_matrix

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(self.ntcrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                # loss = torch.sum(self.custom_function(torch.abs(q-F.softmax(student_out[v],dim=-1)),self.threshold)*(-q) * F.log_softmax(student_out[v], dim=-1), dim=-1)
                loss = torch.sum(torch.exp(-torch.abs(q-F.softmax(student_out[v],dim=-1))) * (-q)*F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


if __name__ == "__main__":
    loss_func = Margin_DINOLoss(
        out_dim = 65536,
        ncrops = 6,
        ntcrops = 2,
        warmup_teacher_temp = 0.04,
        teacher_temp = 0.07,
        warmup_teacher_temp_epochs = 30,
        nepochs = 100,
        threshold = 0.6,
    ).cuda()
    teacher_out = torch.randn(6,65536).cuda()
    student_out = torch.randn(18,65536).cuda()
    loss = loss_func(student_out,teacher_out,16)
    print(loss)