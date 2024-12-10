from copy import deepcopy
from trainer import prepare_inputs

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data

def regularizations(model, trainer):
    if trainer.cfg.TRAIN.FINETUNE.REGULARIZATION == 'ewc':
        ewc_reg = EWC(model, trainer.train_loader, trainer.cfg)
        return ewc_reg

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model: nn.Module, dataloader, cfg):

        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self.name_list = []
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for inputs in self.dataloader:
            enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)

            ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].float()
            dec_zeros = torch.zeros_like(dec_window[:, -self.cfg.DATA.PRED_LEN:, :]).float()
            dec_window = torch.cat([dec_window[:, :self.cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().cuda()

            # model_cfg = getattr(self.cfg.MODEL, self.cfg.MODEL_NAME.upper())
            model_cfg = self.cfg.MODEL

            if model_cfg.output_attention:
                pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp, finetune=False)[0]
            else:
                pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp, finetune=False)

            pred = pred[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:]

            # ! config 넣어주는거랑 상관없이 이렇게 되니까 원하면 수정해라
            loss = F.mse_loss(pred, ground_truth)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad != None:
                    self.name_list.append(n)
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataloader)


        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.name_list:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
