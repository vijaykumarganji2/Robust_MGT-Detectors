import torch



## Code from :  https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
def MMD(x, y, kernel, device):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)



    return torch.mean(XX + YY - 2. * XY)


import logging
logging.basicConfig(level=logging.INFO)  # OPTIONAL

import sys
from functools import reduce

from torch import nn
import torch.distributed as dist

def distributed():
    # return dist.is_available() and dist.is_initialized()
    return False ## only because I want to use one GPU

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from collections import namedtuple
from typing import Any
from torch.utils.tensorboard import SummaryWriter


class ProjectionMLP(nn.Module):
    '''
    Code for Projection MLP: edit and clean as needed

    Model to project [CLS] representation onto
    another space, where the contrastive loss will
    be calculated.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 300))

    def forward(self, input_features):
        x = input_features[:, 0, :]
        return self.layers(x)


## SimCLR style contrastive loss

class SimCLRContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        try:
            denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        except RuntimeError as e:
            print("DEBUG:")
            print(e)
            # print(self.negatives_mask.shape)
            # print(similarity_matrix.shape)
            # print(self.temperature.shape)
            exit()

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class ContrastivelyInstructedRoberta(torch.nn.Module):
    """
    Wrapper class for model with dict/list rvalues.
    """

    def __init__(self, model: torch.nn.Module, mlp: torch.nn.Module, loss_type: str, logger: SummaryWriter, device: str, lambda_w:float) -> None:
        """
        Init call.
        """
        super().__init__()
        self.model = model
        self.mlp = mlp
        self.loss_type = loss_type
        self.logger = logger
        self.device = device
        self.lambda_w = lambda_w

    def forward(self, src_texts:torch.Tensor, src_masks:torch.Tensor, src_texts_perturb:torch.Tensor, src_masks_perturb:torch.Tensor, \
        tgt_texts:torch.Tensor, tgt_masks:torch.Tensor, tgt_texts_perturb:torch.Tensor, tgt_masks_perturb:torch.Tensor, src_labels:torch.Tensor, tgt_labels:torch.Tensor,\
        fake_labels:torch.Tensor,gen_texts:torch.Tensor,gen_src_masks:torch.Tensor,gen_texts_perturb:torch.Tensor,gen_src_masks_perturb:torch.Tensor) -> Any:
        """
        Wrap forward call.
        """
        batch_size = src_texts.shape[0]
        # generator
        src_texts_new = torch.cat([src_texts, gen_texts])
        src_masks_new = torch.cat([src_masks,gen_src_masks])

        src_texts_perturb_new = torch.cat([src_texts_perturb, gen_texts_perturb])
        src_masks_perturb_new = torch.cat([src_masks,gen_src_masks_perturb])

        src_labels_new = torch.cat([src_labels, fake_labels])

        # source
        src_output_dic = self.model(src_texts, attention_mask=src_masks, labels=src_labels)
        src_LCE_real, src_logits_real = src_output_dic["loss"], src_output_dic["logits"]

        src_output_dic_perturbed = self.model(src_texts_perturb, attention_mask=src_masks_perturb, labels=src_labels)
        src_LCE_perturb, src_logits_perturb = src_output_dic_perturbed["loss"], src_output_dic_perturbed["logits"]

        #LCE
        src_output_dic_new = self.model(src_texts_new, attention_mask=src_masks_new, labels=src_labels_new)
        src_LCE_real = src_output_dic_new["loss"]

        src_output_dic_perturbed_new = self.model(src_texts_perturb_new, attention_mask=src_masks_perturb_new, labels=src_labels_new)
        src_LCE_perturb = src_output_dic_perturbed_new["loss"]

        # target
        tgt_output_dic = self.model(tgt_texts, attention_mask=tgt_masks, labels=tgt_labels)
        tgt_LCE_real, tgt_logits_real = tgt_output_dic["loss"], tgt_output_dic["logits"]

        tgt_output_dic_perturbed = self.model(tgt_texts_perturb, attention_mask=tgt_masks_perturb, labels=tgt_labels)
        tgt_LCE_perturb, tgt_logits_perturb = tgt_output_dic_perturbed["loss"], tgt_output_dic_perturbed["logits"]


        # Contrastive losses (simclr supported now)

        if self.loss_type == "simclr":
            ctr_loss = SimCLRContrastiveLoss(batch_size=batch_size)
            ctr_loss.to(self.device)


        if self.loss_type == "simclr":
            src_z_i = self.mlp(src_output_dic["last_hidden_state"])  ## clean
            src_z_j = self.mlp(src_output_dic_perturbed["last_hidden_state"])  ## perturbed
            src_lctr = ctr_loss(src_z_i, src_z_j)
            tgt_z_i = self.mlp(tgt_output_dic["last_hidden_state"])  ## clean
            tgt_z_j = self.mlp(tgt_output_dic_perturbed["last_hidden_state"])  ## perturbed
            tgt_lctr = ctr_loss(tgt_z_i, tgt_z_j)


        ## full loss:

        mmd = MMD(src_z_i, tgt_z_i, kernel='rbf', device=self.device)

        use_ce_perturb = True  ## change for ablations only
        use_both_ce_losses = True  ## change for ablations only
        lambda_mmd = 1.0  ## change for ablations only

        if not use_both_ce_losses:
            loss = self.lambda_w*(src_lctr+tgt_lctr)/2 + lambda_mmd*mmd
        else:
            if use_ce_perturb:
                loss = (1-self.lambda_w)*(src_LCE_real + src_LCE_perturb)/2 \
                        + self.lambda_w*(src_lctr+tgt_lctr)/2 + lambda_mmd*mmd ## final loss used in paper
            else:
                loss = (1-self.lambda_w)*src_LCE_real + self.lambda_w*(src_lctr+tgt_lctr)/2 + lambda_mmd*mmd


        data = {"total_loss":loss, "src_ctr_loss":src_lctr, "tgt_ctr_loss":tgt_lctr, "src_ce_loss_real":src_LCE_real,\
            "src_ce_loss_perturb":src_LCE_perturb,"mmd": mmd, "src_logits":src_logits_real, "tgt_logits":tgt_logits_real}
        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))
            data = data_named_tuple(**data)

        elif isinstance(data, list):
            data = tuple(data)
        return data
    


## all of the training script:
"""Training code for the detector model"""
import pickle
import argparse
import os
import subprocess
from itertools import count
from multiprocessing import Process

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from tqdm.auto import tqdm
from itertools import cycle
from transformers import RobertaTokenizer, tokenization_utils


import sys
sys.path.insert(0,os.getcwd())


from data_loader import Corpus, EncodedDataset, EncodedFakeDataset
from models import RobertaForContrastiveClassification, T5Paraphraser, TextRLReplayBuffer


# torch.manual_seed(int(1000))

DISTRIBUTED_FLAG = False


def setup_distributed(port=29500):
    if not DISTRIBUTED_FLAG:
        return 0, 1

    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return 0, 1

    if 'MPIR_CVAR_CH3_INTERFACE_HOSTNAME' in os.environ:
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        mpi_size = MPI.COMM_WORLD.Get_size()

        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(backend="nccl", world_size=mpi_size, rank=mpi_rank)
        return mpi_rank, mpi_size

    dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size()


def load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, batch_size,
                  max_sequence_length, random_sequence_length):

    real_corpus = Corpus(real_dataset, data_dir=data_dir)
    fake_corpus = Corpus(fake_dataset, data_dir=data_dir)

    real_train, real_valid = real_corpus.train, real_corpus.valid
    real_train_perturb, real_valid_perturb = real_corpus.train_perturb, real_corpus.valid_perturb

    fake_train, fake_valid = fake_corpus.train, fake_corpus.valid
    fake_train_perturb, fake_valid_perturb = fake_corpus.train_perturb, fake_corpus.valid_perturb

    Sampler = DistributedSampler if distributed() and dist.get_world_size() > 1 else RandomSampler

    min_sequence_length = 10 if random_sequence_length else None
    train_dataset = EncodedDataset(real_train, real_train_perturb,fake_train, fake_train_perturb, tokenizer, max_sequence_length, min_sequence_length)
    train_loader = DataLoader(train_dataset, batch_size, sampler=Sampler(train_dataset), num_workers=0, drop_last=True)

    validation_dataset = EncodedDataset(real_valid, real_valid_perturb, fake_valid, fake_valid_perturb, tokenizer, max_sequence_length, min_sequence_length)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=Sampler(validation_dataset))

    fake_dataset = EncodedFakeDataset(fake_train, fake_train_perturb, tokenizer, max_sequence_length, min_sequence_length) #
    fake_loader = DataLoader(fake_dataset, batch_size, sampler=Sampler(fake_dataset), num_workers=0, drop_last=True) #

    return train_loader, validation_loader, fake_loader


def accuracy_sum(logits, labels):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


import math
def train(buffer, model: nn.Module, generator: nn.Module, mlp: nn.Module, loss_type: str, optimizer, generator_optimizer, device: str,
          src_loader: DataLoader,fake_loader:DataLoader, tgt_loader: DataLoader, summary_writer: SummaryWriter, fixed_batch_size, max_sequence_length,
          desc='Train', lambda_w=0.5, ppo_epsilon = 0.2, gamma = 0.01, ppo_buffer_size = 128):

  @torch.no_grad()
  def collect_samples(batch):
    src_data, tgt_data, fake_data = batch
    fake_text_ids = fake_data[0]
    fake_text_perturb_ids = fake_data[2]
    fake_labels = fake_data[4]


    # get text form of both fake_texts
    fake_text = generator.get_text_from_roberta_tokenizer(fake_text_ids)
    fake_text_perturb = generator.get_text_from_roberta_tokenizer(fake_text_perturb_ids)


    #get input_ids using t5 tokenizer
    fake_text_t5_ids = generator.tokenizer(fake_text, return_tensors="pt", max_length = max_sequence_length, truncation=True, padding=True).input_ids
    fake_text_perturb_t5_ids = generator.tokenizer(fake_text_perturb, return_tensors="pt", max_length = max_sequence_length, truncation=True, padding=True).input_ids

    #generate text using generator
    fake_text_t5_ids, fake_text_perturb_t5_ids = fake_text_t5_ids.to(device), fake_text_perturb_t5_ids.to(device)
    gen_text_result = generator.generate_text(fake_text_t5_ids)
    gen_text_perturb_result = generator.generate_text(fake_text_perturb_t5_ids)

    gen_text = gen_text_result['paraphrase_text']
    gen_text_t5_ids = gen_text_result['paraphrase_text_id']
    gen_text_log_probs = gen_text_result['paraphrase_log_prob']

    gen_text_perturb = gen_text_perturb_result['paraphrase_text']
    gen_text_perturb_t5_ids = gen_text_perturb_result['paraphrase_text_id']
    gen_text_perturb_log_probs = gen_text_perturb_result['paraphrase_log_prob']

    #get input_ids using roberta tokenizer
    gen_text_ids, gen_mask = generator.get_roberta_token_ids_from_text(gen_text)
    gen_text_perturb_ids, gen_perturb_mask = generator.get_roberta_token_ids_from_text(gen_text_perturb)


    gen_text_ids, gen_mask, fake_labels = gen_text_ids.to(device), gen_mask.to(device), fake_labels.to(device)
    # model.model --> roberta_model
    gen_output_dic = model.model(gen_text_ids,attention_mask=gen_mask, labels=fake_labels)
    gen_logit_text = gen_output_dic['logits']

    gen_text_perturb_ids, gen_perturb_mask = gen_text_perturb_ids.to(device), gen_perturb_mask.to(device)
    gen_output_dic_perturb = model.model(gen_text_perturb_ids,attention_mask=gen_perturb_mask, labels=fake_labels)
    gen_logit_text_perturb = gen_output_dic_perturb['logits']
    
    #logit is already softmaxxed by model, no need to softmax again..
    #gen_prob_text = F.softmax(gen_logit_text, dim=1)
    #gen_prob_text_perturb = F.softmax(gen_logit_text_perturb, dim=1)
    gen_prob_text = gen_logit_text
    gen_prob_text_perturb = gen_logit_text_perturb

    #rewards
    confidence_gen_text = gen_prob_text[:, 1].tolist()
    confidence_gen_text_perturb = gen_prob_text_perturb[:, 1].tolist()

    #store in proper format
    fake_data.append(fake_text_t5_ids)
    fake_data.append(fake_text_perturb_t5_ids)

    gen_data = [gen_text, gen_text_ids, gen_mask, gen_text_t5_ids, gen_text_log_probs, confidence_gen_text]
    gen_perturb_data = [gen_text_perturb, gen_text_perturb_ids, gen_perturb_mask, gen_text_perturb_t5_ids, gen_text_perturb_log_probs, confidence_gen_text_perturb]

    #store the current batch in buffer.
    buffer.update_batch(src_data = src_data, tgt_data = tgt_data, fake_data = fake_data, gen_data = gen_data, gen_perturb_data = gen_perturb_data)

    batch_size = src_data[0].shape[0]
    buffer.size += batch_size
    # print(buffer.size)
    # print('Complete One Batch')


  model.train()
  generator.train()

  src_train_accuracy = 0
  tgt_train_accuracy = 0
  train_epoch_size = 0
  train_loss = 0
  # train_iteration = 0
  gen_loss = 0
  # total = len(src_loader)
  epoch_generator_loss = []
  epoch_model_loss = []

  #fill the buffer
  loader_len = None
  if len(src_loader)==len(tgt_loader):
      triple_loader = enumerate(zip(src_loader, tgt_loader,cycle(fake_loader)))
      loader_len = len(src_loader)
  elif len(src_loader)<len(tgt_loader):
      print("Src smaller than Tgt")
      triple_loader = enumerate(zip(cycle(src_loader),tgt_loader,cycle(fake_loader)))
      loader_len = len(tgt_loader)
  else:
      triple_loader = enumerate(zip(src_loader,cycle(tgt_loader),cycle(fake_loader)))
      loader_len = len(src_loader)

  # print(fixed_batch_size)
  expected_num_rollout = math.ceil(loader_len*fixed_batch_size/ ppo_buffer_size)

  pbar = tqdm(total = expected_num_rollout)
  pbar.set_description(f'{desc}, Rollouts ')
  rollout = 1
  epoch_end = False
  i=0
  while True:
    if epoch_end:
        break

    # pbar.set_description(f'{desc} : Rollout {rollout} [Buffer Filling]')
    buffer_pbar = tqdm(total = ppo_buffer_size, leave = False)
    buffer_pbar.set_description(f'Rollout {rollout} [Filling Buffer]')
    while buffer.size < ppo_buffer_size:
      try:
        idx, batch = next(triple_loader)
        collect_samples(batch)
        buffer_pbar.update(batch[0][0].shape[0])
      except StopIteration:
        epoch_end = True
        break
      # pbar.set_postfix({'buffer_size': buffer.size})
      buffer_pbar.set_postfix({'buffer_size': buffer.size})
    buffer_pbar.close()


    if buffer.size == 0:
      continue

    # Generator Training
    # generator.train()
    # print('Generator Training', desc)
    # pbar.set_description(f'{desc} : Rollout {rollout} [Generator Training]')
    buffer_generator_loss = []
    with tqdm(buffer.get_buffer(), desc = f'Rollout {rollout} [Generator Training]', leave = False) as loop:

      for mini_batch in loop:
        src_data, tgt_data, fake_data, gen_data, gen_perturb_data = mini_batch

        batch_size = src_data[0].shape[0]

        gen_text_t5_ids, gen_text_old_log_prob, gen_text_rewards = (gen_data[3]).to(device), (gen_data[4]).to(device), torch.from_numpy(gen_data[-1]).to(device)
        fake_text_t5_ids = (fake_data[5]).to(device)

        gen_text_log_prob = generator(fake_text_t5_ids, gen_text_t5_ids)['log_probs']

        ratio = (gen_text_log_prob - gen_text_old_log_prob).exp()
        policy_loss1 = gen_text_rewards * ratio
        policy_loss2 = gen_text_rewards * torch.clamp(ratio, 1.0 - ppo_epsilon, 1.0 + ppo_epsilon)
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
        diversity = (-gen_text_log_prob * gen_text_log_prob.exp()).sum()
        loss = policy_loss - gamma*diversity

        #loss for perturb data
        gen_text_perturb_t5_ids, gen_text_perturb_old_log_prob, gen_text_perturb_rewards = (gen_perturb_data[3]).to(device), (gen_perturb_data[4]).to(device), torch.from_numpy(gen_perturb_data[-1]).to(device)
        fake_text_perturb_t5_ids = (fake_data[6]).to(device)
        gen_text_perturb_log_prob = generator(fake_text_perturb_t5_ids, gen_text_perturb_t5_ids)['log_probs']

        ratio_perturb = (gen_text_perturb_log_prob - gen_text_perturb_old_log_prob).exp()
        policy_loss1_perturb = gen_text_perturb_rewards * ratio_perturb
        policy_loss2_perturb = gen_text_perturb_rewards * torch.clamp(ratio_perturb, 1.0 - ppo_epsilon, 1.0 + ppo_epsilon)
        policy_loss_perturb = -torch.min(policy_loss1_perturb, policy_loss2_perturb).mean()
        diversity_perturb = (-gen_text_perturb_log_prob * gen_text_perturb_log_prob.exp()).sum()
        loss_perturb = policy_loss_perturb - gamma*diversity_perturb


        generator_loss = (loss+loss_perturb)/2.0

        #backward step
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        gen_loss += generator_loss.item()*batch_size

        postfix_str = f"Generator Loss={generator_loss.item()}"    

        loop.set_postfix(gen_loss = generator_loss.item())
        buffer_generator_loss.append(generator_loss.item())

    epoch_generator_loss.append(buffer_generator_loss)


    #model Training
    # model.train()
    # i=0
    buffer_model_loss = []
    with tqdm(buffer.get_buffer(), desc = f'Rollout {rollout} [Model Training]', leave = False) as loop:
      for mini_batch in loop:
        src_data, tgt_data, fake_data, gen_data, gen_perturb_data = mini_batch

        src_texts, src_masks, src_texts_perturb, src_masks_perturb, src_labels = src_data[0], src_data[1], src_data[2], src_data[3], src_data[4]
        src_texts, src_masks, src_labels = src_texts.to(device), src_masks.to(device), src_labels.to(device)
        src_texts_perturb, src_masks_perturb = src_texts_perturb.to(device), src_masks_perturb.to(device)
        batch_size = src_texts.shape[0]

        tgt_texts, tgt_masks, tgt_texts_perturb, tgt_masks_perturb, tgt_labels = tgt_data[0], tgt_data[1], tgt_data[2], tgt_data[3], tgt_data[4]
        tgt_texts, tgt_masks, tgt_labels = tgt_texts.to(device), tgt_masks.to(device), tgt_labels.to(device)
        tgt_texts_perturb, tgt_masks_perturb = tgt_texts_perturb.to(device), tgt_masks_perturb.to(device)

        fake_texts, fake_masks, fake_texts_perturb, fake_masks_perturb, fake_labels = fake_data[0], fake_data[1], fake_data[2], fake_data[3], fake_data[4]
        fake_texts, fake_masks, fake_labels = fake_texts.to(device), fake_masks.to(device), fake_labels.to(device)
        fake_texts_perturb, fake_masks_perturb = fake_texts_perturb.to(device), fake_masks_perturb.to(device)
        gen_texts, gen_src_masks = gen_data[1], gen_data[2]
        gen_texts_perturb, gen_src_masks_perturb = gen_perturb_data[1], gen_perturb_data[2]


        output_dic = model(src_texts, src_masks, src_texts_perturb, src_masks_perturb,\
        tgt_texts, tgt_masks, tgt_texts_perturb, tgt_masks_perturb,\
        src_labels, tgt_labels,fake_labels,gen_texts,gen_src_masks,gen_texts_perturb,gen_src_masks_perturb)

        loss = output_dic.total_loss

        #backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        src_batch_accuracy = accuracy_sum(output_dic.src_logits, src_labels)
        src_train_accuracy += src_batch_accuracy
        tgt_batch_accuracy = accuracy_sum(output_dic.tgt_logits, tgt_labels)
        tgt_train_accuracy += tgt_batch_accuracy
        train_epoch_size += batch_size
        train_loss += loss.item() * batch_size

        loop.set_postfix(loss=loss.item(),src_acc=src_train_accuracy / train_epoch_size,
                         tgt_acc=tgt_train_accuracy / train_epoch_size,
                         mmd=output_dic.mmd.item(),
                         src_LCE_real=output_dic.src_ce_loss_real.item(), src_LCE_perturb=output_dic.src_ce_loss_perturb.item())

        buffer_model_loss.append(loss.item())
    epoch_model_loss.append(buffer_model_loss)

    buffer.clear()
    rollout+=1
    pbar.update(1)

  pbar.close()

  return {
        "train/src_accuracy": src_train_accuracy,
        "train/tgt_accuracy": tgt_train_accuracy,
        "train/epoch_size": train_epoch_size,
        "generator/loss": gen_loss,
        "train/loss": train_loss
    }, {'epoch_generator_loss' : epoch_generator_loss, 'epoch_model_loss' : epoch_model_loss}

def validate(model: nn.Module, device: str, loader: DataLoader, votes=1, desc='Validation'):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    records = [record for v in range(votes) for record in tqdm(loader, desc=f'Preloading data ... {v}',
                                                               disable=distributed() and dist.get_rank() > 0)]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm(records, desc=desc, disable=distributed() and dist.get_rank() > 0) as loop, torch.no_grad():
        for example in loop:
            losses = []
            logit_votes = []

            for texts, masks, texts_perturb, masks_perturb, labels in example:
                texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
                batch_size = texts.shape[0]

                output_dic = model(texts, attention_mask=masks, labels=labels)
                loss, logits = output_dic["loss"], output_dic["logits"]
                losses.append(loss)
                logit_votes.append(logits)

            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)

            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)

    return {
        "validation/accuracy": validation_accuracy,
        "validation/epoch_size": validation_epoch_size,
        "validation/loss": validation_loss
    }


def _all_reduce_dict(d, device):
    # wrap in tensor and use reduce to gpu0 tensor
    output_d = {}
    for (key, value) in sorted(d.items()):
        tensor_input = torch.tensor([[value]]).to(device)
        # torch.distributed.all_reduce(tensor_input)
        output_d[key] = tensor_input.item()
    return output_d


def run(src_data_dir,
        tgt_data_dir,
        src_real_dataset,
        src_fake_dataset,
        tgt_real_dataset,
        tgt_fake_dataset,
        model_save_path,
        model_save_name,
        batch_size,
        loss_type,
        max_epochs=10,
        device=None,
        max_sequence_length=512,
        max_buffer_size = 128,
        random_sequence_length=False,
        epoch_size=None,
        seed=None,
        token_dropout=None,
        large=False,
        learning_rate=2e-5,
        weight_decay=0,
        load_from_checkpoint=False,
        lambda_w=0.5,
        checkpoint_name='',
        **kwargs):
    args = locals()
    rank, world_size = setup_distributed()

    if device is None:
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
        # device = f'cuda:1' if torch.cuda.is_available() else 'cpu'

    #if device=='cpu':
    #    print("Could not find GPU")
    #    exit()

    print('rank:', rank, 'world_size:', world_size, 'device:', device)

    logdir = os.environ.get("OPENAI_LOGDIR", "logs/conda_gan")
    os.makedirs(logdir, exist_ok=True)

    writer = SummaryWriter(logdir) if rank == 0 else None

    # import torch.distributed as dist
    if distributed() and rank > 0:
        dist.barrier()

    model_name = 'roberta-large' if large else 'roberta-base'
    tokenization_utils.logger.setLevel('ERROR')


    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaForContrastiveClassification.from_pretrained('roberta-base').to(device)

    mlp = ProjectionMLP().to(device)

    generator = T5Paraphraser(roberta_tokenizer=tokenizer, max_sequence_length = max_sequence_length).to(device)

    model = ContrastivelyInstructedRoberta(model=roberta_model,mlp=mlp, loss_type=loss_type, logger=writer, device=device, lambda_w=lambda_w)

    buffer = TextRLReplayBuffer(max_buffer_size = max_buffer_size)
    
    # generator_model = GeneratorModel(roberta_model,generator).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator_optimizer = Adam(generator.parameters(), lr=learning_rate)
    

    start_epoch = 1
    best_validation_accuracy = 0

    if load_from_checkpoint:
        checkpoint_path = os.path.join(model_save_path, checkpoint_name)
        generator_checkpoint_path = os.path.join(model_save_path, 'generator_'+checkpoint_name)
        if os.path.exists(checkpoint_path) and os.path.exists(generator_checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            roberta_model.load_state_dict(checkpoint['model_state_dict'])

            model = ContrastivelyInstructedRoberta(model=roberta_model, mlp=mlp, loss_type=loss_type, logger=writer, device=device, lambda_w=lambda_w)
            optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            combined_metrics = checkpoint['metrics_state_dict']
            best_validation_accuracy = combined_metrics["validation/accuracy"]

            generator_checkpoint = torch.load(generator_checkpoint_path)
            generator.load_state_dict(generator_checkpoint['generator_state_dict'])
            generator_optimizer = Adam(generator.parameters(), lr=learning_rate)
            generator_optimizer.load_state_dict(generator_checkpoint['optimizer_state_dict'])

            print(f">>>>>>> Resuming training from epoch {start_epoch} <<<<<<<<<<<<<")
       


    if rank == 0:
        # summary(model)
        if distributed():
            dist.barrier()

    if world_size > 1:
        model = DistributedDataParallel(model, [rank], output_device=rank, find_unused_parameters=True)


    src_train_loader, src_validation_loader,src_fake_loader = load_datasets(src_data_dir, src_real_dataset, src_fake_dataset, tokenizer, batch_size,
                                                    max_sequence_length, random_sequence_length)

    tgt_train_loader, tgt_validation_loader,_ = load_datasets(tgt_data_dir, tgt_real_dataset, tgt_fake_dataset, tokenizer, batch_size,
                                                    max_sequence_length, random_sequence_length)
    print(len(src_train_loader), len(tgt_train_loader))

   
    epoch_loop = range(start_epoch,2) if max_epochs is None else range(start_epoch, max_epochs + 1)
    
    without_progress = 0
    earlystop_epochs = 10
    epoch_wise_losses = []
    for epoch in epoch_loop:
        if world_size > 1:
            src_train_loader.sampler.set_epoch(epoch)
            src_validation_loader.sampler.set_epoch(epoch)
            tgt_train_loader.sampler.set_epoch(epoch)
            tgt_validation_loader.sampler.set_epoch(epoch)

        train_metrics, train_losses = train(buffer, model,generator, mlp, loss_type,optimizer,generator_optimizer, device,
                              src_train_loader,src_fake_loader, tgt_train_loader, writer, batch_size, max_sequence_length,
                              f'Epoch {epoch}', lambda_w=lambda_w)

        validation_metrics = validate(roberta_model, device, src_validation_loader) ## we are only using supervision on the source

        combined_metrics = _all_reduce_dict({**validation_metrics, **train_metrics}, device)

        combined_metrics["train/src_accuracy"] /= combined_metrics["train/epoch_size"]
        combined_metrics["train/loss"] /= combined_metrics["train/epoch_size"]
        combined_metrics["validation/accuracy"] /= combined_metrics["validation/epoch_size"]
        combined_metrics["validation/loss"] /= combined_metrics["validation/epoch_size"]

        combined_metrics["generator/loss"] /= combined_metrics["train/epoch_size"]
        combined_metrics["train/tgt_accuracy"] /= combined_metrics["train/epoch_size"]

        epoch_wise_losses.append(train_losses)

        if rank == 0:
            #save losses in pickle format for future reference
            with open(os.path.join(model_save_path, model_save_name.split('.')[0]+'_losses.pkl'), 'wb') as f:
              pickle.dump(epoch_wise_losses, f)

            for key, value in combined_metrics.items():
                writer.add_scalar(key, value, global_step=epoch)

            model_to_save = roberta_model.module if hasattr(roberta_model, 'module') else roberta_model
            torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model_to_save.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        metrics_state_dict=combined_metrics,
                        args=args
                    ),
                    os.path.join(model_save_path, checkpoint_name)
                )
            torch.save(dict(generator_state_dict=generator.state_dict(),
                                optimizer_state_dict = generator_optimizer.state_dict()
                                ),os.path.join(model_save_path, 'generator_'+checkpoint_name))
            
            print(f"Checkpoint saved for epoch {epoch}")

            if combined_metrics["validation/accuracy"] > best_validation_accuracy:
                without_progress = 0
                best_validation_accuracy = combined_metrics["validation/accuracy"]

                model_to_save = roberta_model.module if hasattr(roberta_model, 'module') else roberta_model
                torch.save(dict(
                        epoch=epoch,
                        model_state_dict=model_to_save.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        metrics_state_dict=combined_metrics,
                        args=args
                    ),
                    os.path.join(model_save_path, model_save_name)
                )
                torch.save(dict(generator_state_dict=generator.state_dict(),
                                optimizer_state_dict = generator_optimizer.state_dict()
                                ),os.path.join(model_save_path, 'generator_'+model_save_name))
                print(">>>>>>>>>>  best Model saved for :", epoch, "<<<<<<<<<<<<<<<<<<<<<<")
        without_progress += 1

        if without_progress >= earlystop_epochs:
            break
    
    print(">>>>>>>>>>>>> Training Completed <<<<<<<<<<<<<<")



os.chdir('/home/tarun/MTP')


RANDOM_SEED = 123

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


#src_data_dir = './data/RODAM_Data/OVR_Dataset/OPT30B_Version/wp_OVR/source_data'
#tgt_data_dir = './data/RODAM_Data/OVR_Dataset/OPT30B_Version/wp_OVR/target_data'
src_data_dir = './data/RODAM_Data/medquad_gpt_setup/train_valid_source_test_target/source_data'
tgt_data_dir = './data/RODAM_Data/medquad_gpt_setup/train_valid_source_test_target/target_data'
src_real_dataset = 'real'
src_fake_dataset = 'fake'
tgt_real_dataset = 'real'
tgt_fake_dataset = 'fake'
model_save_path = './Trained_Models/RODAM/medquad'
model_save_name = f'rodam_train_valid_source_test_target_medquad_gpt.pt'
checkpoint_name = f'checkpoint_rodam_train_valid_source_test_target_medquad_gpt.pt'
batch_size = 8 # decrease batch size if gpu memory overflowed
epochs = 10
device = 1
loss_type = 'simclr'
# seed = 42

run(src_data_dir=src_data_dir,
    tgt_data_dir=tgt_data_dir,
    src_real_dataset=src_real_dataset,
    src_fake_dataset=src_fake_dataset,
    tgt_real_dataset=tgt_real_dataset,
    tgt_fake_dataset=tgt_fake_dataset,
    model_save_path=model_save_path,
    model_save_name=model_save_name,
    batch_size=batch_size,
    loss_type=loss_type,
    max_epochs=epochs,
    device= device,
    load_from_checkpoint=False,
    checkpoint_name=checkpoint_name)    

