from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

def _gather_subfeat(feat, ind, c=None):
    """Gather a subset of features at the 2nd dimension

    Args:
        feat (b, N, d):                 The input feature 
        ind (b, N, c) or (b, N):        The gather index or starting index
        c (int):                        The consecutive number if the ind is the starting index
    
    Returns:
        feat (b, N, c):                 The gathered sub feature
    """
    b, N, _ = feat.shape
    # Generate the index if the 
    if len(ind.shape) == 2:
        assert c is not None
        ind_complete = torch.arange(c, device=ind.device).repeat(b, N, 1)
        ind_complete = ind_complete + ind.unsqueeze(2)
        ind = ind_complete
    
    # gather
    feat = feat.gather(2, ind)
    return feat

# feat = torch.randn(1, 2, 4)
# ind = torch.tensor([[0, 2]])
# c = 2
# print(feat)
# print(ind)
# print(c)
# print(_gather_subfeat(feat, ind, c))



def _gather_feat(feat, ind, mask=None):
	"""Gather feature from the 1st dimension

	Args:
		feat (b, N, c): The input feature.
		ind (b, K ): The gather index. k is the gather number. (e.g. the gathered pixel number if N is the # of pixels h*w)
		mask (_type_, optional): _description_. Defaults to None.

	Returns:
		feat (b, K, c) or (N_masked, c) if mask is not None
	"""
	dim = feat.size(2)
	ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
	feat = feat.gather(1, ind)
	if mask is not None:
		mask = mask.unsqueeze(2).expand_as(feat)
		feat = feat[mask]
		feat = feat.view(-1, dim)
	return feat


def _transpose_and_gather_feat(feat, ind):
    """Gather the feature at the indexed locations ((h, w) maps)

    Args:
        feat (b, c, h, w): 
        ind (b, K): K is the number of elements to be gathered.

    Returns:
		feat (b, K, c)
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()	# (b, h, w, c)
    feat = feat.view(feat.size(0), -1, feat.size(3)) # (b, h*w, c)
    feat = _gather_feat(feat, ind)
    return feat


def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)


def flip_lr(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def flip_lr_off(x, flip_idx):
    tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    shape = tmp.shape
    tmp = tmp.reshape(tmp.shape[0], 17, 2,
                      tmp.shape[2], tmp.shape[3])
    tmp[:, :, 0, :, :] *= -1
    for e in flip_idx:
        tmp[:, e[0], ...], tmp[:, e[1], ...] = \
            tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
    return torch.from_numpy(tmp.reshape(shape)).to(x.device)


def _inds_to_coords(inds, w):
    """indices to coordinates given the image width

    Args:
        inds (any shape): _description_
        w (_type_): _description_
    
    Returns:
        coords: the inds shape with a 2 appended at the last element for the x, y coordinates in the OpenCV coordinate
    """
    ys = (inds / w).int().float()
    xs = (inds % w).int().float()
    coords = torch.cat(
      (xs.unsqueeze(-1), ys.unsqueeze(-1)),
      dim=-1
    )

    return coords
