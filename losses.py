import torch
import torch.nn.functional as F

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
    

class MarginLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(MarginLoss, self).__init__()
        self.margin = margin

    def forward(self, scores, labels):
        correct_scores = scores[labels == 1]
        incorrect_scores = scores[labels == 0]
        losses = torch.relu(self.margin - correct_scores + incorrect_scores)
        return losses.mean()

class TripletLossCosine(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLossCosine, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = 1 - F.cosine_similarity(anchor, positive)
        distance_negative = 1 - F.cosine_similarity(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class MixedTripletLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, margin=1.0, margin_cosine=0.2):
        super(MixedTripletLoss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.margin_cosine = margin_cosine

    def forward(self, anchor, positive, negative):
        triplet_loss = TripletLoss(self.margin)
        triplet_loss_cosine = TripletLossCosine(self.margin_cosine)
        
        loss_triplet = triplet_loss(anchor, positive, negative)
        loss_triplet_cosine = triplet_loss_cosine(anchor, positive, negative)
        
        mixed_loss = loss_triplet + self.alpha * (loss_triplet_cosine)
        
        return mixed_loss





class L1Loss(torch.nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, input, target):
        return F.l1_loss(input, target)

class L2Loss(torch.nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, input, target):
        return F.mse_loss(input, target)
    

       
#Ref:https://github.com/andreasveit/triplet-network-pytorch/
class TripletLossWithRegularization(torch.nn.Module):
    def __init__(self, margin=1.0, regularization_weight=0.001):
        super(TripletLossWithRegularization, self).__init__()
        self.margin = margin
        self.regularization_weight = regularization_weight

    def forward(self, embedded_x, embedded_y, embedded_z):
       
        loss_wrapper = torch.nn.MarginRankingLoss()
        dista = F.pairwise_distance(embedded_x, embedded_y, p=2)
        distb = F.pairwise_distance(embedded_x, embedded_z, p=2)
        
        target = torch.FloatTensor(dista.size()).fill_(1)
        if dista.is_cuda:
            target = target.cuda()
        target = torch.autograd.Variable(target)
        loss_triplet = loss_wrapper(dista, distb, target)
        
        loss_embedd = embedded_x.norm(
            2) + embedded_y.norm(2) + embedded_z.norm(2)

        loss = loss_triplet + self.regularization_weight * loss_embedd
        return loss
