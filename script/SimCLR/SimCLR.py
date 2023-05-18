import torch
import torch.nn as nn


# Define the SimCLR model without a projection head
# https://arxiv.org/abs/2002.05709
class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, temperature=0.07, status='train'):
        super(SimCLRModel, self).__init__()
        self.base_encoder = base_encoder
        self.temperature = temperature
        self.status = status

    def forward(self, x):

        if self.status == 'train':
            # Pass the inputs through the base encoder
            h_i = self.base_encoder(x[0])
            h_j = self.base_encoder(x[1])

            # Apply L2 normalization to the projections (optional)
            z_i = nn.functional.normalize(h_i, dim=1)
            z_j = nn.functional.normalize(h_j, dim=1)

            # Compute the NT-Xent loss
            loss = self.nt_xent_loss(z_i, z_j, self.temperature)

            return loss

        elif self.status == 'test':
            # Pass the inputs through the base encoder
            h_i = self.base_encoder(x)

            # Apply L2 normalization to the projections (optional)
            z_i = nn.functional.normalize(h_i, dim=1)

            return z_i

    def nt_xent_loss(self, z_i, z_j, temperature):
        # Concatenate the representations of the anchor and positive examples
        representations = torch.cat([z_i, z_j], dim=0)

        # https://arxiv.org/abs/2208.06530 (Equation 1)
        # Euclidean distance
        euclidean_distance = torch.cdist(representations, representations)
        # print(euclidean_distance)
        # print(torch.cdist(representations[:,None,:], representations[None,:,:]))
        euclidean_similarity = 1 / (1 + euclidean_distance)
        # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
        # Exclude self-similarity (main diagonal = 0)
        self_mask = torch.eye(euclidean_similarity.shape[0], dtype=torch.bool, device=euclidean_similarity.device)
        euclidean_similarity.masked_fill_(self_mask, -9e15)

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=euclidean_similarity.shape[0] // 2, dims=0)
        # InfoNCE loss
        euclidean_sim = euclidean_similarity / temperature
        nll = -euclidean_sim[pos_mask] + torch.logsumexp(euclidean_sim, dim=-1)
        nll = nll.mean()

        return nll
