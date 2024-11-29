import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

class AFA(nn.Module):
    def __init__(self, dim, num_heads):
        super(AFA, self).__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.mhattn = nn.MultiheadAttention(dim, num_heads=num_heads)

    def forward(self, stu, tea):
        assert stu.shape == tea.shape, f"student's shape must be equal to teacher's shape..."
        B, C, H, W = stu.shape
        N = H * W
        stu = stu.flatten(start_dim=2).transpose(1, 2)  # [B, C, H, W] -> [B, C, N] -> [B, N, C]
        tea = tea.flatten(start_dim=2).transpose(1, 2)

        stu_qkv = self.qkv(stu).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        tea_qkv = self.qkv(tea).reshape(B, N, 3, C).permute(2, 0, 1, 3)

        q_s, k_s, v_s = stu_qkv.unbind(0)
        q_t, k_t, v_t = tea_qkv.unbind(0)

        output_s, atten_weight_s = self.mhattn(q_s, k_t, v_t)
        output_t, atten_weight_t = self.mhattn(q_t, k_s, v_s)

        loss = F.l1_loss(atten_weight_s, atten_weight_t) + F.l1_loss(output_s, output_t)
        return loss

class WFD(nn.Module):
    def __init__(self):
        super(WFD, self).__init__()
        self.xfm = DWTForward(J=3, mode='zero', wave='haar')
        self.alpha, self.beta = 0.3, 0.7

    def forward(self, stu, tea):
        cA_s, (cH_s, cV_s, cD_s) = self.xfm(stu)
        cA_t, (cH_t, cV_t, cD_t) = self.xfm(tea)

        cH_s, cH_t = cH_s.sum(2), cH_t.sum(2)
        cV_s, cV_t = cV_s.sum(2), cV_t.sum(2)
        cD_s, cD_t = cD_s.sum(2), cD_t.sum(2)

        low_freq_loss = F.l1_loss(cA_s, cA_t)
        high_freq_loss = F.l1_loss(cH_s, cH_t) + F.l1_loss(cV_s, cV_t) + F.l1_loss(cD_s, cD_t)
        return self.alpha * high_freq_loss + self.beta * low_freq_loss

class TotalLoss(nn.Module):
    def __init__(self, dims, heads=[4, 4, 8, 8]):
        super(TotalLoss, self).__init__()
        self.dims = dims

        for i in range(len(dims)):
            afa = AFA(dims[i], heads[i])
            self.__setattr__(f'afa_{i}', afa)

            wfd = WFD()
            self.__setattr__(f'wfd_{i}', wfd)

    def forward(self, students, teachers):
        assert len(students) == len(teachers), 'The number of teachers and students features not equal...'
        n = len(students)
        losses = 0.0

        for i in range(n):
            afa = self.__getattr__(f'afa_{i}')
            wfd = self.__getattr__(f'wfd_{i}')

            afa_loss = afa(students[i], teachers[i])
            wfd_loss = wfd(students[i], teachers[i])

            losses = losses + afa_loss + wfd_loss

        return losses

if __name__ == '__main__':
    x1 = torch.randn(1, 64, 96, 320)
    x2 = torch.randn(1, 128, 48, 160)
    x3 = torch.randn(1, 256, 24, 80)
    x4 = torch.randn(1, 512, 12, 40)

    stu = [x1, x2, x3, x4]
    tea = [x1, x2, x3, x4]

    m = TotalLoss([64, 128, 256, 512])
    outs = m(stu, tea)
    print(outs)