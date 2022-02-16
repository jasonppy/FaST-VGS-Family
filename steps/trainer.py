import time
import os
import torch
import math
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datasets import spokencoco_dataset, places_dataset, flickr8k_dataset, libri_dataset
from datasets.sampler import StatefulSampler
from models import fast_vgs
from .utils import *
from .trainer_utils import *
from .bert_adam import BertAdam
from apex.fp16_utils import *
from apex import amp
from logging import getLogger
logger = getLogger(__name__)

class Trainer:
    @staticmethod
    def add_args(parser):
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--exp_dir", type=str)
        parser.add_argument("--trained_weights_dir", type=str, default=None)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--val_batch_size", type=int)
        parser.add_argument("--val_cross_batch_size", type=int)
        parser.add_argument("--n_epochs", type=int)
        parser.add_argument("--n_print_steps", type=int)
        parser.add_argument("--n_val_steps", type=int)
        parser.add_argument("--lr", type=float, default=0.0001)
        parser.add_argument("--warmup_fraction", type=float, default=0.1)
    
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"number of devices: {torch.cuda.device_count()}")
        self.writer = SummaryWriter(self.args.exp_dir)
        self.seed_everything(seed=self.args.seed)
        self.meters = self._setup_meters()
        self.progress, self.total_progress = setup_progress(self)
        self.dual_encoder, self.cross_encoder, self.trainables, self.indices, self.libri_indices, self.optim_states = self._setup_models()
        self.use_libri_loss = self.args.libri_w2v2_weight != None
        self.train_loader, self.valid_loader, self.valid_loader2, self.train_sampler, self.libri_train_loader, self.libri_valid_loader, self.libri_train_sampler, self.train_data_length = self._setup_dataloader()
        self.total_num_updates = int(math.floor(self.train_data_length / self.args.batch_size))*self.args.n_epochs
        self.optimizer = self._setup_optimizer()
        if torch.cuda.device_count() > 1:
            self.dual_encoder = nn.DataParallel(self.dual_encoder)
            self.cross_encoder = nn.DataParallel(self.cross_encoder)
        self.scheduler = self._setup_scheduler()
        self.criterion = fast_vgs.Margin_InfoNCE_loss
        logger.info(f"batch size: {self.args.batch_size}")
    
    def forward(self, batch):
        audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls, losses = self.dual_encoder(audio_feats = batch['audio'], attention_mask = batch['audio_attention_mask'], visual_feats = batch['visual_feats'], visual_pos = batch['visual_pos'], target_list = batch['label'])
        coarse_cross_relationship_score_matrix = visual_cls @ audio_cls.transpose(0,1)
        losses['coarse_matching_loss'] = fast_vgs.Margin_InfoNCE_loss(coarse_cross_relationship_score_matrix, margin=self.args.margin, img_id = batch['img_id'])
        B = visual_feats.shape[0]
        visual_feats_square = visual_feats.repeat(B,1,1)
        audio_feats_square = audio_feats.repeat_interleave(B, dim=0)
        extended_audio_attention_mask_square = extended_audio_attention_mask.repeat_interleave(B, dim=0)
        cross_relationship_score_square = self.cross_encoder(audio_feats_square, extended_audio_attention_mask_square, visual_feats_square)
        cross_relationship_score_matrix = cross_relationship_score_square.view(B,B)
        losses["fine_matching_loss"] = fast_vgs.Margin_InfoNCE_loss(cross_relationship_score_matrix, margin=self.args.margin, img_id = batch['img_id'])
        return losses

    def train(self):
        flag = True
        step_per_epoch = int(self.train_data_length/self.args.batch_size)
        data_start_time = time.time()

        while flag:
            if self.use_libri_loss:
                libri_loader_iterator = iter(self.libri_train_loader)
            for i, batch in enumerate(self.train_loader):
                if self.use_libri_loss:
                    libri_batch = next(libri_loader_iterator)
                data_end_time = time.time()
                self.dual_encoder.train()
                self.cross_encoder.train()
                if self.progress['num_updates'] > self.total_num_updates:
                    flag = False
                    self.validate_and_save()
                    self.writer.close()
                    break
                
                cur_lr = np.mean(self.optimizer.get_lr())

                self.writer.add_scalar("lr", cur_lr, self.progress['num_updates'])
                cur_step = self.progress['num_updates'] % step_per_epoch

                cur_batch = {
                        "visual_feats": batch['visual_feats'].to(self.device),
                        "visual_pos": batch['boxes'].to(self.device),
                        "audio": batch['audio'].to(self.device),
                        "audio_attention_mask": batch['audio_attention_mask'].to(self.device),
                        "img_id": batch['img_id'],
                        "label": batch['label']
                        }

                losses = self.forward(cur_batch)
                if self.use_libri_loss:
                    losses.update(self.dual_encoder(audio_feats = libri_batch['audio'].to(self.device), attention_mask = libri_batch['audio_attention_mask'].to(self.device), target_list = libri_batch['label'], forward_libri=True))

                for key in losses:
                    if key in self.meters:
                        self.meters[key].update(losses[key].mean().cpu().item(), cur_batch['visual_feats'].shape[0])
                        self.writer.add_scalar(key, self.meters[key].val, self.progress['num_updates'])
                
                weighted_loss = self.weight_loss(losses)

                self.meters['weighted_loss'].update(weighted_loss.item(), cur_batch['visual_feats'].shape[0])
                self.writer.add_scalar('weighted_loss', weighted_loss.item(), self.progress['num_updates'])

                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainables, 1.)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.meters['data_time'].update(data_end_time - data_start_time)
                self.meters['train_time'].update(time.time() - data_end_time)
                #########
                self.writer.add_scalar("data_time", data_end_time - data_start_time, self.progress['num_updates'])
                self.writer.add_scalar("train_time", time.time() - data_end_time, self.progress['num_updates'])

                # logging
                if self.progress['num_updates'] % self.args.n_print_steps == 0:
                    log_out = {}
                    log_out['epoch'] = f"{self.progress['epoch']}/{self.args.n_epochs}"
                    log_out['cur_step/steps_per_epoch'] = f"{cur_step}/{step_per_epoch}"
                    log_out['num_updates'] = self.progress['num_updates']
                    log_out['lr'] = f"{cur_lr:.7f}"
                    for key in self.meters:
                        if self.meters[key].val != 0 or self.meters[key].avg != 0:
                            log_out[key] = f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})" if isinstance(self.meters[key].val, float) else f"{self.meters[key].val}"
                    logger.info(log_out)
                    if np.isnan(self.meters['weighted_loss'].avg):
                        logger.info("training diverged...")
                        return
                # validation and save models
                if self.progress['num_updates'] % self.args.n_val_steps == 0:
                    self.validate_and_save(libri=self.use_libri_loss, places=self.args.places)
                self.progress['num_updates'] += 1
                self.progress['epoch'] = int(math.ceil(self.progress['num_updates'] / step_per_epoch))
                data_start_time = time.time()

    def validate_and_save(self, libri=False, places=False):
        self.dual_encoder.eval()
        self.cross_encoder.eval()
        if places:
            r10, r5, r1 = self.validate(self.valid_loader)
            r10_unseen, r5_unseen, r1_unseen = self.validate(self.valid_loader2, unseen=True)
            r10, r5, r1 = (r10+r10_unseen)/2, (r5+r5_unseen)/2, (r1+r1_unseen)/2
        else:
            r10, r5, r1 = self.validate_one_to_many()
        
        if libri:
            self.validate_libri()
        # r1 = 0.1 # ignore validation, for debugging
        if r1 > self.progress['best_acc']:
            self.progress['best_epoch'] = self.progress['epoch']
            self.progress['best_acc'] = r1
            save_path = os.path.join(self.args.exp_dir,"best_bundle.pth")
            torch.save(
                {
                    "dual_encoder": self.dual_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.dual_encoder.state_dict(),
                    "cross_encoder": self.cross_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.cross_encoder.state_dict(),
                    "optimizer":  self.optimizer.state_dict(),
                    "indices": self.train_sampler.state_dict(),
                    "libri_indices": self.libri_train_sampler.state_dict() if self.libri_train_sampler is not None else None
                },save_path
            )
            logger.info(f"save *best* models at {save_path} at global step {self.progress['num_updates']}")
        save_progress(self)
        save_path = os.path.join(self.args.exp_dir,"bundle.pth")
        torch.save(
            {
                "dual_encoder": self.dual_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.dual_encoder.state_dict(),
                "cross_encoder": self.cross_encoder.module.state_dict() if torch.cuda.device_count() > 1 else self.cross_encoder.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "indices": self.train_sampler.state_dict(),
                "libri_indices": self.libri_train_sampler.state_dict() if self.libri_train_sampler is not None else None
            },save_path
        )
        logger.info(f"save models, indices, acc and other statistics at {save_path} and {self.args.exp_dir}/progress.pkl at global step {self.progress['num_updates']}")

    def validate(self, valid_loader, unseen = False):
        start_val_time = time.time()
        N_examples = self.valid_loader.dataset.__len__()

        # frame_counts = []
        with torch.no_grad():
            # get single modal representations
            audio_feats_total = [] 
            extended_audio_attention_mask_total = []
            visual_feats_total = [] 
            img_id_total = []
            audio_cls_total = []
            visual_cls_total = []
            for i, batch in enumerate(valid_loader):
                self.dual_encoder.eval()
                self.cross_encoder.eval()
                audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls = self.dual_encoder(audio_feats = batch['audio'].to(self.device), attention_mask = batch['audio_attention_mask'].to(self.device), visual_feats = batch['visual_feats'].to(self.device), visual_pos = batch['boxes'].to(self.device), test = True)
                audio_cls_total.append(audio_cls)
                visual_cls_total.append(visual_cls)
                audio_feats_total.append(audio_feats.detach()) # still on cude after .detach(), just removed from graph, so no gradient
                extended_audio_attention_mask_total.append(extended_audio_attention_mask.detach())
                visual_feats_total.append(visual_feats.detach())
                img_id_total.append(batch['img_id'])

            audio_feats_total = torch.cat(audio_feats_total)
            extended_audio_attention_mask_total = torch.cat(extended_audio_attention_mask_total)
            visual_feats_total = torch.cat(visual_feats_total)
            img_id_total = np.concatenate(img_id_total)

            visual_cls_total = torch.cat(visual_cls_total)
            audio_cls_total = torch.cat(audio_cls_total)
            coarse_cross_relationship_score_matrix = audio_cls_total @ visual_cls_total.transpose(0,1)
            recalls = calc_recalls_from_S_coarse(coarse_cross_relationship_score_matrix, img_id=img_id_total)
            avg_acc_coarse = (recalls['A_r10'] + recalls['I_r10']) / 2
            avg_acc_r1_coarse = (recalls['A_r1'] + recalls['I_r1']) / 2
            self.writer.add_scalar("acc_coarse", avg_acc_coarse, self.progress['num_updates'])
            self.writer.add_scalar("acc_r1_coarse", avg_acc_r1_coarse, self.progress['num_updates'])
            
            logger.info("Coarse Retrieval Accuracy:" if not unseen else "Coarse Retrieval Accuracy (Unseen):")
            logger.info('Audio R@100 {A_r100:.3f} Image R@100 {I_r100:.3f} Average R@100 {r100_ave:.3f} over {N:d} validation pairs'.format(A_r100=recalls['A_r100'], I_r100=recalls['I_r100'], r100_ave=(recalls['A_r100']+recalls['I_r100'])/2, N=N_examples))
            logger.info('Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} Average R@10 {r10_ave:.3f} over {N:d} validation pairs'.format(A_r10=recalls['A_r10'], I_r10=recalls['I_r10'], r10_ave=(recalls['A_r10']+recalls['I_r10'])/2, N=N_examples))
            logger.info('Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} Average R@5 {r5_ave:.3f} over {N:d} validation pairs'.format(A_r5=recalls['A_r5'], I_r5=recalls['I_r5'], r5_ave=(recalls['A_r5']+recalls['I_r5'])/2, N=N_examples))
            logger.info('Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} Average R@1 {ave_r1:.3f} over {N:d} validation pairs'.format(A_r1=recalls['A_r1'], I_r1=recalls['I_r1'], ave_r1=(recalls['A_r1']+recalls['I_r1'])/2,  N=N_examples))
            logger.info(f"validation time: {time.time() - start_val_time:.3f}")  

            if self.args.fine_matching_weight > 0:
                if self.args.coarse_to_fine_retrieve:
                    # visual indices should be 1000*100 + 100*1000
                    # audio indices should be 100*1000 + 1000*100
                    visual_indices, audio_indices = coarse_retrieve(coarse_cross_relationship_score_matrix, topk=self.args.topk)
                    B = len(visual_indices)
                    val_cross_batch_size = self.args.val_cross_batch_size
                    num_steps = math.ceil(B / val_cross_batch_size)
                else:
                    # logger.info(f"total num of pairs {B**2}, val cross batch size: {val_cross_batch_size}, num of steps: {num_steps}")
                    # get O(B^2) pairs
                    # # original audio image pair: (1,a) (2,b) (3,c)
                    # # image: [a,b,c,d] -> [a,b,c,a,b,c,a,b,c]
                    # # audio: [1,2,3,4] -> [1,1,1,2,2,2,3,3,3]
                    # # to avoid unnecessary duplication, we just repeat indices
                    B = visual_feats_total.shape[0]
                    val_cross_batch_size = self.args.val_cross_batch_size
                    num_steps = math.ceil(B**2 / val_cross_batch_size)
                    visual_indices = torch.LongTensor(list(range(B))).repeat(B)
                    audio_indices = torch.LongTensor(list(range(B))).repeat_interleave(B,dim=0)

                # get cross modal representations
                cross_relationship_score_square = []
                for i in range(num_steps):
                    visual_feats_square = visual_feats_total[visual_indices[i*val_cross_batch_size:(i+1)*val_cross_batch_size]].to(self.device)
                    audio_feats_square = audio_feats_total[audio_indices[i*val_cross_batch_size:(i+1)*val_cross_batch_size]].to(self.device)
                    extended_audio_attention_mask_square = extended_audio_attention_mask_total[audio_indices[i*val_cross_batch_size:(i+1)*val_cross_batch_size]].to(self.device)
                    cross_relationship_score = self.cross_encoder(audio_feats_square, extended_audio_attention_mask_square, visual_feats_square)
                    # logger.info(f"shape of cross_relationship_score {cross_relationship_score.shape}")
                    cross_relationship_score_square.append(cross_relationship_score.detach())

                # do not test visual encoder ability here, might consider doing it in the future
                # visual_feats = visual_feats_square[::(B+1)][:,1:]
                cross_relationship_score_square = torch.cat(cross_relationship_score_square)
                # logger.info(f"validation cross relationship score data type: {cross_relationship_score_square.dtype}")
                if self.args.coarse_to_fine_retrieve:
                    recalls = fine_retrieve(cross_relationship_score_square, anchor_img_id=img_id_total, visual_indices = visual_indices, audio_indices = audio_indices, topk=self.args.topk, B = B)
                else:
                    cross_relationship_score_matrix = cross_relationship_score_square.view(B,B)
                    recalls = calc_recalls_from_S(cross_relationship_score_matrix, img_id=img_id_total)

                logger.info("Fine Retrieval Accuracy:" if not unseen else "Fine Retrieval Accuracy (Unseen):")
                logger.info('Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} Average R@10 {r10_ave:.3f} over {N:d} validation pairs'.format(A_r10=recalls['A_r10'], I_r10=recalls['I_r10'], r10_ave=(recalls['A_r10']+recalls['I_r10'])/2, N=N_examples))
                logger.info('Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} Average R@5 {r5_ave:.3f} over {N:d} validation pairs'.format(A_r5=recalls['A_r5'], I_r5=recalls['I_r5'], r5_ave=(recalls['A_r5']+recalls['I_r5'])/2, N=N_examples))
                logger.info('Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} Average R@1 {ave_r1:.3f} over {N:d} validation pairs'.format(A_r1=recalls['A_r1'], I_r1=recalls['I_r1'], ave_r1=(recalls['A_r1']+recalls['I_r1'])/2,  N=N_examples))  

                logger.info(f"validation time: {time.time() - start_val_time:.3f}")  

        avg_acc_r10 = (recalls['A_r10'] + recalls['I_r10']) / 2
        avg_acc_r5 = (recalls['A_r5'] + recalls['I_r5']) / 2
        avg_acc_r1 = (recalls['A_r1'] + recalls['I_r1']) / 2
        if unseen:
            self.writer.add_scalar("acc_r10_unseen", avg_acc_r10, self.progress['num_updates'])
            self.writer.add_scalar("acc_r5_unseen", avg_acc_r5, self.progress['num_updates'])
            self.writer.add_scalar("acc_r1_unseen", avg_acc_r1, self.progress['num_updates'])
        else:
            self.writer.add_scalar("acc_r10", avg_acc_r10, self.progress['num_updates'])
            self.writer.add_scalar("acc_r5", avg_acc_r5, self.progress['num_updates'])
            self.writer.add_scalar("acc_r1", avg_acc_r1, self.progress['num_updates'])
        return avg_acc_r10, avg_acc_r5, avg_acc_r1

    def validate_one_to_many(self, hide_progress=True):
        self.dual_encoder.eval()
        self.cross_encoder.eval()
        N_examples = self.valid_loader.dataset.__len__()

        with torch.no_grad():
            # get single modal representations
            audio_feats_total = [] 
            extended_audio_attention_mask_total = []
            audio_cls_total = []
            audio_img_id_total = [] # this is same order as audio_cls_total and audio_feats_total
            img_id_to_img_feats = {}
            img_img_id_list = []
            img_cls_list = [] # this is distinct, order is the same as img_img_id_list
            img_feats_list = [] # this is distinct, order is the same as img_img_id_list
            for i, batch in enumerate(self.valid_loader):
                self.dual_encoder.eval()
                self.cross_encoder.eval()
                
                audio_feats, audio_cls, extended_audio_attention_mask, visual_feats, visual_cls = self.dual_encoder(audio_feats = batch['audio'].to(self.device), attention_mask = batch['audio_attention_mask'].to(self.device), visual_feats = batch['visual_feats'].to(self.device), visual_pos = batch['boxes'].to(self.device), test = True)
                audio_cls_total.append(audio_cls)
                # visual_cls_total.append(visual_cls)
                audio_feats_total.append(audio_feats.detach()) # still on cude after .detach(), just removed from graph, so no gradient
                extended_audio_attention_mask_total.append(extended_audio_attention_mask.detach())
                # visual_feats_total.append(visual_feats.detach())
                detached_visual_feats = visual_feats.detach()
                audio_img_id_total.append(batch['img_id'])
                for i, img_id in enumerate(batch['img_id']):
                    if img_id not in img_id_to_img_feats:
                        img_id_to_img_feats[img_id] = detached_visual_feats[i]
                        img_feats_list.append(detached_visual_feats[i])
                        img_cls_list.append(visual_cls[i].detach())
                        img_img_id_list.append(img_id)

            audio_cls_total = torch.cat(audio_cls_total)
            img_cls_list = torch.stack(img_cls_list)
            img_feats_list = torch.stack(img_feats_list)
            audio_feats_total = torch.cat(audio_feats_total)
            extended_audio_attention_mask_total = torch.cat(extended_audio_attention_mask_total)
            audio_img_id_total = np.concatenate(audio_img_id_total)
            img_img_id_list = np.array(img_img_id_list)

            coarse_cross_relationship_score_matrix = img_cls_list @ audio_cls_total.transpose(0,1)
            recalls = calc_recalls_from_S_one_to_many_coarse(coarse_cross_relationship_score_matrix, row_img_id=img_img_id_list, column_img_id=audio_img_id_total)
            avg_acc_coarse = (recalls['A_r10'] + recalls['I_r10']) / 2
            avg_acc_r1_coarse = (recalls['A_r1'] + recalls['I_r1']) / 2
            self.writer.add_scalar("acc_coarse", avg_acc_coarse, self.progress['num_updates'])
            self.writer.add_scalar("acc_r1_coarse", avg_acc_r1_coarse, self.progress['num_updates'])
            logger.info("Coarse Retrieval Accuracy:")
            logger.info('Audio R@100 {A_r100:.3f} Image R@100 {I_r100:.3f} Average R@100 {r100_ave:.3f} over {N:d} validation pairs'.format(A_r100=recalls['A_r100'], I_r100=recalls['I_r100'], r100_ave=(recalls['A_r100']+recalls['I_r100'])/2, N=N_examples))
            logger.info('Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} Average R@10 {r10_ave:.3f} over {N:d} validation pairs'.format(A_r10=recalls['A_r10'], I_r10=recalls['I_r10'], r10_ave=(recalls['A_r10']+recalls['I_r10'])/2, N=N_examples))
            logger.info('Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} Average R@5 {r5_ave:.3f} over {N:d} validation pairs'.format(A_r5=recalls['A_r5'], I_r5=recalls['I_r5'], r5_ave=(recalls['A_r5']+recalls['I_r5'])/2, N=N_examples))
            logger.info('Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} Average R@1 {ave_r1:.3f} over {N:d} validation pairs'.format(A_r1=recalls['A_r1'], I_r1=recalls['I_r1'], ave_r1=(recalls['A_r1']+recalls['I_r1'])/2,  N=N_examples))
            if self.args.coarse_to_fine_retrieve:
                visual_indices, audio_indices = coarse_retrieve_one_to_many(coarse_cross_relationship_score_matrix.transpose(0,1), topk=self.args.topk) # transpose to have [audio_len, visual_len]
                B = len(visual_indices)
                val_cross_batch_size = self.args.val_cross_batch_size
                num_steps = math.ceil(B / val_cross_batch_size)
                cross_relationship_score_square = []
                for i in tqdm(range(num_steps), disable=hide_progress):
                    visual_feats_square = img_feats_list[visual_indices[i*val_cross_batch_size:(i+1)*val_cross_batch_size]].to(self.device)
                    audio_feats_square = audio_feats_total[audio_indices[i*val_cross_batch_size:(i+1)*val_cross_batch_size]].to(self.device)
                    extended_audio_attention_mask_square = extended_audio_attention_mask_total[audio_indices[i*val_cross_batch_size:(i+1)*val_cross_batch_size]].to(self.device)
                    cross_relationship_score = self.cross_encoder(audio_feats_square, extended_audio_attention_mask_square, visual_feats_square, extended_visual_attention_mask_square=None)
                    cross_relationship_score_square.append(cross_relationship_score.detach())

                # do not test visual encoder ability here, might consider doing it in the future
                # visual_feats = visual_feats_square[::(B+1)][:,1:]
                cross_relationship_score_square = torch.cat(cross_relationship_score_square)
            # logger.info(f"validation cross relationship score data type: {cross_relationship_score_square.dtype}")
                recalls = fine_retrieve_one_to_many(cross_relationship_score_square, audio_img_id=audio_img_id_total, visual_img_id=img_img_id_list, visual_indices = visual_indices, audio_indices = audio_indices, topk=self.args.topk)
            else:
                cross_relationship_score_matrix = torch.zeros((len(img_img_id_list), audio_feats_total.shape[0])).to(self.device)
                for i, img_id in enumerate(tqdm(img_img_id_list, disable=hide_progress)):
                    visual_feats_cur = img_id_to_img_feats[img_id].unsqueeze(0).repeat(audio_feats_total.shape[0],1,1)
                    # B = audio_feats_total.shape[0]
                    temp = self.cross_encoder(audio_feats_total, extended_audio_attention_mask_total, visual_feats_cur, None)
                    cross_relationship_score_matrix[i,:] = temp.squeeze(1)
                recalls = calc_recalls_from_S_one_to_many(cross_relationship_score_matrix, row_img_id=img_img_id_list, column_img_id=audio_img_id_total)

        logger.info("Fine Retrieval Accuracy:")
        logger.info('Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} Average R@10 {r10_ave:.3f} over {N:d} validation pairs'.format(A_r10=recalls['A_r10'], I_r10=recalls['I_r10'], r10_ave=(recalls['A_r10']+recalls['I_r10'])/2, N=N_examples))
        logger.info('Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} Average R@5 {r5_ave:.3f} over {N:d} validation pairs'.format(A_r5=recalls['A_r5'], I_r5=recalls['I_r5'], r5_ave=(recalls['A_r5']+recalls['I_r5'])/2, N=N_examples))
        logger.info('Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} Average R@1 {ave_r1:.3f} over {N:d} validation pairs'.format(A_r1=recalls['A_r1'], I_r1=recalls['I_r1'], ave_r1=(recalls['A_r1']+recalls['I_r1'])/2,  N=N_examples))

        avg_acc_r10 = (recalls['A_r10'] + recalls['I_r10']) / 2
        avg_acc_r5 = (recalls['A_r5'] + recalls['I_r5']) / 2
        avg_acc_r1 = (recalls['A_r1'] + recalls['I_r1']) / 2
        self.writer.add_scalar("acc_r10", avg_acc_r10, self.progress['num_updates'])
        self.writer.add_scalar("acc_r5", avg_acc_r5, self.progress['num_updates'])
        self.writer.add_scalar("acc_r1", avg_acc_r1, self.progress['num_updates'])
        return avg_acc_r10, avg_acc_r5, avg_acc_r1
        
    def validate_libri(self):
        with torch.no_grad():
            N = 0
            total_loss = 0
            for batch in self.libri_valid_loader:
                self.dual_encoder.eval()
                n = len(batch['audio'])
                N += n
                losses = self.dual_encoder(audio_feats = batch['audio'].to(self.device), attention_mask = batch['audio_attention_mask'].to(self.device), target_list=batch['label'], forward_libri=True)
                total_loss += losses['libri_w2v2_loss'].mean()*n
        cur_val_loss = (total_loss/N).item()
        self.writer.add_scalar("libri_val_loss", cur_val_loss, self.progress['num_updates'])

        if cur_val_loss < self.progress['best_libri_val_loss']:
            self.progress['best_libri_val_loss'] = cur_val_loss
            logger.info(f"libri validation loss: {cur_val_loss:.3f}*\n")
        else:
            logger.info(f"libri validation loss: {cur_val_loss:.3f}\n")

    def _setup_meters(self):
        meters = {}
        meter_names = ['weighted_loss', "fine_matching_loss", "coarse_matching_loss", 'caption_w2v2_loss', "libri_w2v2_loss", "caption_hubert_loss", "libri_hubert_loss", "caption_m_acc", "libri_m_acc",'data_time', 'train_time']
        for name in meter_names:
            meters[name] = AverageMeter()
        return meters
    
    def _setup_models(self):
        dual_encoder = fast_vgs.DualEncoder(self.args)
        cross_encoder = fast_vgs.CrossEncoder(self.args)
        print_model_info(dual_encoder)
        print_model_info(cross_encoder)
        if self.args.trained_weights_dir != None:
            bundle = torch.load(os.path.join(self.args.trained_weights_dir, "best_bundle.pth"))
            dual_encoder.carefully_load_state_dict(bundle['dual_encoder'])
            cross_encoder.carefully_load_state_dict(bundle['cross_encoder'])
            indices = None
            libri_indices = None
            optim_states = None
            # logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
            logger.info(f"Load trained weights from {self.args.trained_weights_dir}")
        elif self.args.validate:
            bundle = torch.load(os.path.join(self.args.exp_dir, "best_bundle.pth"))
            dual_encoder.carefully_load_state_dict(bundle['dual_encoder'])
            cross_encoder.carefully_load_state_dict(bundle['cross_encoder'])
            indices = None
            libri_indices = None
            optim_states = None
            # logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
            logger.info("Perform Validation")
        elif self.progress['num_updates'] > 1:
            bundle = torch.load(os.path.join(self.args.exp_dir, "bundle.pth"))
            dual_encoder.carefully_load_state_dict(bundle['dual_encoder'])
            cross_encoder.carefully_load_state_dict(bundle['cross_encoder'])
            indices = bundle['indices']
            libri_indices = bundle['libri_indices']
            optim_states = bundle['optimizer']
            logger.info("loaded parameters and data indices from epoch %d, global step %d" % (self.progress['epoch'], self.progress['num_updates']))
        else:
            indices = None
            libri_indices = None
            optim_states = None

        if self.args.fb_w2v2_weights_fn and self.progress['num_updates'] <= 1 and not self.args.validate and self.args.trained_weights_dir == None:
            dual_encoder.conv1_trm1_trm3.carefully_load_state_dict(torch.load(self.args.fb_w2v2_weights_fn)['model'])

        if self.args.feature_grad_mult <= 0.:
            for name, p in dual_encoder.named_parameters():
                if "feature_extractor" in name:
                    p.requires_grad = False
        trainables1 = [p for p in dual_encoder.parameters() if p.requires_grad]
        trainables2 = [p for p in cross_encoder.parameters() if p.requires_grad]
        trainables = trainables1 + trainables2

        dual_encoder.to(self.device)
        cross_encoder.to(self.device)

        return dual_encoder, cross_encoder, trainables, indices, libri_indices, optim_states

    def _setup_dataloader(self):
        if self.args.places:
            # raise NotImplementedError
            train_dataset = places_dataset.ImageCaptionDataset(self.args, split='train')
            val_seen_dataset = places_dataset.ImageCaptionDataset(self.args, split='val_seen')
            val_unseen_dataset = places_dataset.ImageCaptionDataset(self.args, split='val_unseen')
            train_sampler = StatefulSampler(len(train_dataset))
            if self.progress['num_updates'] > 1 and self.indices is not None:
                train_sampler.load_state_dict(self.indices)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=True)
            valid_loader = torch.utils.data.DataLoader(val_seen_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_seen_dataset.collate)
            valid_loader2 = torch.utils.data.DataLoader(val_unseen_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_unseen_dataset.collate)
        elif self.args.flickr8k:
            train_dataset = flickr8k_dataset.ImageCaptionDataset(self.args, split='train')
            val_dataset = flickr8k_dataset.ImageCaptionDataset(self.args, split='val')
            train_sampler = StatefulSampler(len(train_dataset))
            if self.progress['num_updates'] > 1 and self.indices is not None:
                train_sampler.load_state_dict(self.indices)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=True)
            valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_dataset.collate)
            valid_loader2 = None
        else:
        # SpokenCOCO
            train_dataset = spokencoco_dataset.ImageCaptionDataset(self.args, split='train')
            val_dataset = spokencoco_dataset.ImageCaptionDataset(self.args, split='val')
            train_sampler = StatefulSampler(len(train_dataset))
            if self.progress['num_updates'] > 1 and self.indices is not None:
                train_sampler.load_state_dict(self.indices)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=True, sampler = train_sampler, collate_fn = train_dataset.collate, drop_last=True)
            valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.args.val_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True, collate_fn = val_dataset.collate)
            valid_loader2 = None

        if self.use_libri_loss:
            # librispeech dataloaders
            # train
            step_per_epoch = int(np.floor(len(train_dataset)/self.args.batch_size))
            # libri_train_dataset = libri_dataset_mm.LibriDataset(self.args, split="train")
            libri_train_dataset = libri_dataset.LibriDataset(self.args, split="train")
            libri_train_bzs = libri_train_dataset.calculate_batch_size(step_per_epoch)
            libri_train_bzs = min(libri_train_bzs, 64)
            logger.info(f"librispeech train batch size: {libri_train_bzs}")
            libri_train_sampler = StatefulSampler(len(libri_train_dataset))
            if self.progress['num_updates'] > 1 and self.libri_indices is not None:
                libri_train_sampler.load_state_dict(self.libri_indices)
            libri_train_loader = torch.utils.data.DataLoader(libri_train_dataset, batch_size=libri_train_bzs, num_workers=self.args.num_workers, pin_memory=True, sampler = libri_train_sampler, collate_fn = libri_train_dataset.collate, drop_last=True)
            
            # val
            # libri_val_dataset = libri_dataset_mm.LibriDataset(self.args, split="val")
            libri_val_dataset = libri_dataset.LibriDataset(self.args, split="val")
            logger.info(f"librispeech val batch size: {self.args.libri_val_bzs}")
            libri_valid_loader = torch.utils.data.DataLoader(libri_val_dataset, batch_size=self.args.libri_val_bzs, num_workers=self.args.num_workers, pin_memory=True, collate_fn = libri_val_dataset.collate, drop_last=True)
        else:
            libri_train_loader = None
            libri_valid_loader = None
            libri_train_sampler = None

        return train_loader, valid_loader, valid_loader2, train_sampler, libri_train_loader, libri_valid_loader, libri_train_sampler, len(train_dataset)
    
    def _setup_optimizer(self):
        optimizer = BertAdam(self.trainables, lr=self.args.lr, warmup=self.args.warmup_fraction, t_total=self.total_num_updates)

        if self.progress['num_updates'] > 1:
            optimizer.load_state_dict(self.optim_states)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        optimizer.zero_grad()
        return optimizer
    
    def _setup_scheduler(self):
        pass

    def weight_loss(self, losses):
        weighted_loss = losses['coarse_matching_loss'] * self.args.coarse_matching_weight + losses['fine_matching_loss'] * self.args.fine_matching_weight
        if 'caption_w2v2_loss' in losses:
            weighted_loss += losses['caption_w2v2_loss'].mean() * self.args.caption_w2v2_weight
        if 'libri_w2v2_loss' in losses:
            weighted_loss += losses['libri_w2v2_loss'].mean() * self.args.libri_w2v2_weight
        if 'caption_hubert_loss' in losses:
            weighted_loss += losses['caption_hubert_loss'].mean() * self.args.caption_hubert_weight
        if 'libri_hubert_loss' in losses:
            weighted_loss += losses['libri_hubert_loss'].mean() * self.args.libri_hubert_weight
        
        return weighted_loss
    
    def seed_everything(self, seed=1):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

