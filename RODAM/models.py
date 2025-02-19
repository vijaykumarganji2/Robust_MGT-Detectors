import torch
from torch.nn import Softmax
from torch.nn import CrossEntropyLoss, MSELoss
from typing import Optional, Tuple

from transformers import RobertaForSequenceClassification

from transformers.modeling_outputs import SequenceClassifierOutput

from dataclasses import dataclass

@dataclass
class SequenceClassifierOutputWithLastLayer(SequenceClassifierOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaForContrastiveClassification(RobertaForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.soft_max = Softmax(dim=1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        softmax_logits = self.soft_max(logits)

        if not return_dict:
            output = (softmax_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithLastLayer(
            loss=loss,
            logits=softmax_logits,
            last_hidden_state=sequence_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



"""RADAR Generator"""

import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

import numpy as np
from operator import itemgetter

from data_loader import PreProcess

class TextRLReplayBuffer:

    def __init__(self, max_buffer_size=512, momentum=0.90):
        self.max_buffer_size = max_buffer_size
        self.buffer = []
        self.size = 0
        self.momentum = momentum

        self.reward_mean = 0.0
        self.reward_mean_sq = 0.0
        self.reward_std = 1.0

        self.reward_mean_perturb = 0.0
        self.reward_mean_sq_perturb = 0.0
        self.reward_std_perturb = 1.0

    def update_batch(self, src_data, tgt_data, fake_data, gen_data, gen_perturb_data, normalize_reward=True):

      if normalize_reward:
        rewards = np.array(gen_data[-1])
        batch_momentum = self.momentum**len(rewards)
        self.reward_mean = self.reward_mean * batch_momentum + np.mean(rewards) * (1 - batch_momentum)
        self.reward_mean_sq = self.reward_mean_sq * batch_momentum + np.mean(rewards**2) * (1 - batch_momentum)
        self.reward_std = np.abs(self.reward_mean_sq - self.reward_mean**2)**0.5
        normalized_rewards = (rewards - self.reward_mean) / (self.reward_std + 1e-5)
        normalized_rewards = np.clip(normalized_rewards, -2.0, 2.0)
        gen_data.append(normalized_rewards)

        perturb_rewards = np.array(gen_perturb_data[-1])
        batch_momentum = self.momentum**len(perturb_rewards)
        self.reward_mean_perturb = self.reward_mean_perturb * batch_momentum + np.mean(perturb_rewards) * (1 - batch_momentum)
        self.reward_mean_sq_perturb = self.reward_mean_sq_perturb * batch_momentum + np.mean(perturb_rewards**2) * (1 - batch_momentum)
        self.reward_std_perturb = np.abs(self.reward_mean_sq_perturb - self.reward_mean_perturb**2)**0.5
        normalized_rewards_perturb = (perturb_rewards - self.reward_mean_perturb) / (self.reward_std_perturb + 1e-5)
        normalized_rewards_perturb = np.clip(normalized_rewards_perturb, -2.0, 2.0)
        gen_perturb_data.append(normalized_rewards_perturb)

      else:
        gen_data.append(np.array(gen_data[-1]))
        gen_perturb_data.append(np.array(gen_perturb_data[-1]))

      self.buffer.append([src_data, tgt_data, fake_data, gen_data, gen_perturb_data])

    def get_buffer(self):
      return self.buffer

    def clear(self):
        self.buffer = []
        self.size = 0

class T5Paraphraser(nn.Module):
    def __init__(self, roberta_tokenizer, max_sequence_length):
        super(T5Paraphraser, self).__init__()
        self.model_name = "t5-small"
        self.c = T5Config.from_pretrained('t5-small', dropout_rate = 0.0)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, config = self.c)
        self.max_sequence_length = max_sequence_length
        self.preprocessor = PreProcess(special_chars_norm=True, lowercase_norm=True, period_norm=True, proper_norm=True, accented_norm=True)
        self.roberta_tokenizer = roberta_tokenizer

    def forward(self, input_text_ids, labels_ids):
      # input_texts = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
      # labels =  self.tokenizer(paraphrase_text, return_tensors="pt", truncation=True, padding=True).input_ids.to(device)
      # print(type(input_text_ids))
      # print(input_text_ids)
      input_attention_masks = torch.where(input_text_ids!=0 , 1, 0)
      # input_text_ids = input_text_ids.to(device)
      # labels_ids = labels_ids.to(device)
      # print(input_texts.input_ids.shape)
      # print(labels.shape)
      output = self.model(input_ids=input_text_ids,attention_mask=input_attention_masks, labels=labels_ids)

      return {'log_probs' : self.get_log_prob_from_logit(output.logits, labels_ids), 'loss' : output.loss}

    def get_log_prob_from_logit(self, logits, labels):
      output_log_probs = logits.log_softmax(-1)
      output_gen_log_probs = torch.gather(output_log_probs, -1, labels[:, :, None]).squeeze(-1)
      log_p = output_gen_log_probs.sum(dim = -1)
      return log_p/output_gen_log_probs.shape[1]

    def generate_text(self, input_text_ids):

      input_attention_masks = torch.where(input_text_ids!=0 , 1, 0)
      # input_text_ids = input_text_ids.to(device)
      # input_ids = self.tokenizer(input_batch['ai_text'], return_tensors="pt", truncation=True, padding=True).to(device)
      # print(input_ids.input_ids.shape)
      labels = self.model.generate(input_ids=input_text_ids, attention_mask=input_attention_masks, min_length = 30, max_length=self.max_sequence_length,
                                   early_stopping = True, length_penalty = 2.0, no_repeat_ngram_size = 3, num_beams = 4)
      paraphrase_text = self.get_paraphrase_text(labels)
      # labels = labels.to(device)
      # print(labels.shape)
      # print(output)
      output = self.model(input_ids=input_text_ids,attention_mask=input_attention_masks, labels=labels)
      # paraphrase_log_prob = self.compute_log_prob(output.sequences, output.scores)
      result = {'paraphrase_text':paraphrase_text, 'paraphrase_text_id' :labels, 'paraphrase_log_prob' : self.get_log_prob_from_logit(output.logits,labels)}
      return result

    # def compute_log_prob(self, sequences, scores):
    #   gen_sequences = sequences[:, 1:]
    #   log_probs = torch.stack(scores, dim=1).log_softmax(-1)
    #   gen_log_probs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)
    #   log_probs_sum = gen_log_probs.sum(dim = -1)
    #   # print(gen_log_probs.shape[1])
    #   # return log_probs_sum/gen_log_probs.shape[1]
    #   return log_probs_sum

    def get_paraphrase_text(self, output):
      paraphrase_text = []
      for i in range(output.shape[0]):
        paraphrase_text.append(self.tokenizer.decode(output[i], skip_special_tokens=True))
      return paraphrase_text

    def get_text_from_roberta_tokenizer(self, input_ids):
      output_text = []
      for i in range(input_ids.shape[0]):
        output_text.append(self.roberta_tokenizer.decode(input_ids[i], skip_special_tokens=True))
      return output_text

    def get_roberta_token_ids_from_text(self,input_texts):
      paraphrase_ids_list = []
      attention_masks_list = []
      for paraphrase in input_texts:
        processed_paraphrase = self.preprocessor.fit(paraphrase)
        padded_sequence = self.roberta_tokenizer(processed_paraphrase, padding='max_length', max_length=self.max_sequence_length, truncation=True)

        # Append to lists
        paraphrase_ids_list.append(torch.tensor(padded_sequence['input_ids']))
        attention_masks_list.append(torch.tensor(padded_sequence['attention_mask']))

      # Convert lists to tensors
      paraphrase_ids_batch = torch.stack(paraphrase_ids_list, dim=0)
      attention_masks_batch = torch.stack(attention_masks_list, dim=0)
      return paraphrase_ids_batch, attention_masks_batch
