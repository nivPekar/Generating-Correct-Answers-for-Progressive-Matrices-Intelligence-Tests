import torch
import torch.nn





class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=195, with_updating_sum_dist=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sum_negative_dist = 0
        self.sum_positive_dist = 0
        self.with_updating_sum_dist = with_updating_sum_dist

    def restart_sum_dist(self):
        self.sum_negative_dist = 0
        self.sum_positive_dist = 0

    def forward(self, x0, x1, y):

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        #print(y.data[0])
        if self.with_updating_sum_dist:
            if y.data[0] == 1:
                self.sum_positive_dist += dist.cpu().sum().detach().numpy()/dist.size()[0]
            else:
                self.sum_negative_dist += dist.cpu().sum().detach().numpy()/dist.size()[0]

        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]

        return loss