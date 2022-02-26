
import numpy as np
import torch
import torch.nn.functional as F
from .trainer_utils import AverageMeter

def compute_perplexity(onehots):
    avg_probs = torch.mean(onehots, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    return perplexity
    
def coarse_retrieve(S, topk):
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(topk, 1)
    I2A_scores, I2A_ind = S.topk(topk, 0)
    # [0,1,2] -> [0,0,1,1,2,2], [topk, n] -> [topk*n]
    audio_indices = torch.cat([torch.tensor(list(range(n))).repeat_interleave(topk, dim=0), I2A_ind.view(topk*n).cpu()])
    # [n,topk] -> [n*topk], [0,1,2] -> [1,2,3,1,2,3]
    visual_indices = torch.cat([A2I_ind.view(n*topk).cpu(), torch.tensor(list(range(n))).repeat(topk)])
    return visual_indices, audio_indices

def fine_retrieve(S, anchor_img_id, visual_indices, audio_indices, topk, B):
    n = B // 2 // topk
    A2I, I2A = torch.chunk(S,2,dim=0)
    _, A2I_ind = A2I.view(n, topk).topk(10, 1)
    _, I2A_ind = I2A.view(topk, n).topk(10, 0)
    visual_imd_id_indices = torch.chunk(visual_indices,2,dim=0)[0].view(n, topk)
    audio_imd_id_indices = torch.chunk(audio_indices,2,dim=0)[1].view(topk, n)
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    for i in range(n):
        I_foundind = -1
        for ind in range(10):
            if anchor_img_id[i] == anchor_img_id[visual_imd_id_indices[i]][A2I_ind[i, ind]]:
                I_foundind = ind
                break
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    for i in range(n):
        A_foundind = -1
        for ind in range(10):
            if anchor_img_id[i] == anchor_img_id[audio_imd_id_indices[:,i]][I2A_ind[ind, i]]:
                A_foundind = ind
                break
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def coarse_retrieve_one_to_many(S, topk):
    audio_len = S.size(0)
    visual_len = S.size(1)
    assert visual_len < audio_len
    A2I_scores, A2I_ind = S.topk(topk, 1)
    I2A_scores, I2A_ind = S.topk(topk, 0)
    # [0,1,2] -> [0,0,1,1,2,2], [topk, n] -> [topk*n]
    audio_indices = torch.cat([torch.tensor(list(range(audio_len))).repeat_interleave(topk, dim=0), I2A_ind.view(topk*visual_len).cpu()])
    # [n,topk] -> [n*topk], [0,1,2] -> [1,2,3,1,2,3]
    visual_indices = torch.cat([A2I_ind.view(audio_len*topk).cpu(), torch.tensor(list(range(visual_len))).repeat(topk)])
    return visual_indices, audio_indices

def fine_retrieve_one_to_many(S, audio_img_id, visual_img_id, visual_indices, audio_indices, topk):
    audio_len = len(audio_img_id)
    visual_len = len(visual_img_id)
    A2I, I2A = S[:audio_len*topk], S[audio_len*topk:]
    _, A2I_ind = A2I.view(audio_len, topk).topk(10, 1)
    _, I2A_ind = I2A.view(topk, visual_len).topk(10, 0)
    visual_imd_id_indices = visual_indices[:audio_len*topk].view(audio_len, topk)
    audio_imd_id_indices = audio_indices[audio_len*topk:].view(topk, visual_len)
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    for i in range(audio_len):
        I_foundind = -1
        for ind in range(10):
            if audio_img_id[i] == visual_img_id[visual_imd_id_indices[i]][A2I_ind[i, ind]]:
                I_foundind = ind
                break
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    for i in range(visual_len):
        A_foundind = -1
        for ind in range(10):
            if visual_img_id[i] == audio_img_id[audio_imd_id_indices[:,i]][I2A_ind[ind, i]]:
                A_foundind = ind
                break
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def calc_recalls_from_S_coarse(S, img_id):
    # audio is row, image is colum
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(100, 1)
    I2A_scores, I2A_ind = S.topk(100, 0)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    A_r100 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    I_r100 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(100):
            if img_id[A2I_ind[i, ind]] == img_id[i]:
                I_foundind = ind
            if img_id[I2A_ind[ind, i]] == img_id[i]:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)
        # do r100s
        if A_foundind >= 0 and A_foundind < 100:
            A_r100.update(1)
        else:
            A_r100.update(0)
        if I_foundind >= 0 and I_foundind < 100:
            I_r100.update(1)
        else:
            I_r100.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg, 'A_r100':A_r100.avg, 'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg, 'I_r100':I_r100.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls
def calc_recalls_from_S(S, img_id):
    # audio is row, image is colum
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 1)
    I2A_scores, I2A_ind = S.topk(10, 0)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if img_id[A2I_ind[i, ind]] == img_id[i]:
                I_foundind = ind
            if img_id[I2A_ind[ind, i]] == img_id[i]:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls
 
def calc_recalls_from_S_one_to_many(S, row_img_id, column_img_id):
    # image is row, audio is colum
    row = S.size(0)
    column = S.size(1)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A2I_scores, A2I_ind = S.topk(10, 0)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(row):
        A_foundind = -1
        for ind in range(10):
            if row_img_id[i] == column_img_id[I2A_ind[i, ind]]:
                A_foundind = ind
                break
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)

    for i in range(column):
        I_foundind = -1
        for ind in range(10):
            if column_img_id[i] == row_img_id[A2I_ind[ind, i]]:
                I_foundind = ind
                break
        # do r1s
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls   

def calc_recalls_from_S_one_to_many_coarse(S, row_img_id, column_img_id):
    # image is row, audio is colum
    row = S.size(0)
    column = S.size(1)
    I2A_scores, I2A_ind = S.topk(100, 1)
    A2I_scores, A2I_ind = S.topk(100, 0)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    A_r100 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    I_r100 = AverageMeter()
    for i in range(row):
        A_foundind = -1
        for ind in range(100):
            if row_img_id[i] == column_img_id[I2A_ind[i, ind]]:
                A_foundind = ind
                break
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 100:
            A_r100.update(1)
        else:
            A_r100.update(0)

    for i in range(column):
        I_foundind = -1
        for ind in range(100):
            if column_img_id[i] == row_img_id[A2I_ind[ind, i]]:
                I_foundind = ind
                break
        # do r1s
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)
        # do r100s
        if I_foundind >= 0 and I_foundind < 100:
            I_r100.update(1)
        else:
            I_r100.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg, 'A_r100':A_r100.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg, 'I_r100':I_r100.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls   