import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        ncrops,
        ntcrops,
        batch_size,
        init_w=10.0,
        init_b=-5.0,
        temperature=1.0,
        base_temperature=0.07,
    ):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ncrops = ncrops
        self.ntcrops = ntcrops
        self.batch_size = batch_size
        self.mask = torch.eye(self.batch_size,dtype=torch.float32).to(torch.device('cuda'))
        self.nviews = 2
        self.mask = self.mask.repeat(self.nviews, self.nviews)
        self.logits_mask = torch.scatter(
            torch.ones_like(self.mask),
            1,
            torch.arange(self.batch_size * self.nviews).view(-1,1).to(torch.device('cuda')),
            0
        )
        self.mask = self.mask * self.logits_mask
    def __similarity__(self,features):
        similarity_matrix = F.cosine_similarity(features.unsqueeze(-1),features.unsqueeze(-1).transpose(0,2))
        return similarity_matrix

    def Calculates(self,features):
        '''
        calculates contrastive loss for mini batch embeddings
        '''
        similarity_matrix = self.__similarity__(features)
        torch.clamp(self.w, 1e-6)
        similarity_matrix = similarity_matrix * self.w + self.b
        similarity_matrix = torch.div(similarity_matrix,self.temperature)

        # Normalized
        logits_max,_     = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits            = similarity_matrix - logits_max.detach()
        
        # compute infonce loss
        exp_logits        = torch.exp(logits) * self.logits_mask
        log_prob          = logits - torch.log(exp_logits.sum(1, keepdim=True))
        loss              = (self.mask * log_prob).sum(1) / self.mask.sum(1)
        loss              = -(self.temperature / self.base_temperature) * loss
        loss              = loss.view(self.nviews,self.batch_size).mean()

        return loss




    def forward(self,student_output, teacher_output):
        """
        Contrastive loss between embeddings of teacher and student networks.
        just compute the contrastive loss between global view (not the same augment)
        """
        student_out = student_output.chunk(self.ntcrops)
        teacher_out = teacher_output.detach().chunk(self.ntcrops)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = self.Calculates(torch.cat((q,student_out[v]),axis=0))
                total_loss += loss
                n_loss_terms +=1
        total_loss /= n_loss_terms
        return total_loss


if __name__ =="__main__":
    stu_features = torch.randn(18,4).cuda()
    tea_features = torch.randn(6,4).cuda()
    Loss = ContrastiveLoss(ncrops=6,ntcrops=2,batch_size=3).cuda()
    loss = Loss.forward(stu_features,tea_features)
    print(loss)