import copy
import os
import time
from typing import Optional, Tuple, Mapping, Union, List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import math
from torch.optim import Optimizer

import models.optimizer as optim
from datasets.loader import get_train_dataloader, get_val_dataloader, get_test_dataloader
from utils.misc import mkdir
from utils.meters import AverageMeter, ProgressMeter
from utils.compare_cfgs import check_cfgs_same

import yaml
from yacs.config import CfgNode as CN

def load_yacs_config(file_path):
    # Load the YAML config file
    with open(file_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    # Convert the YAML dictionary to a YACS CfgNode
    cfg = CN(yaml_cfg)
    return cfg

class Trainer:
    def __init__(
            self,
            cfg,
            model,
            metric_names: Tuple[str],
            loss_names: Tuple[str],
            ft_loss_names: Tuple[str],
            optimizer: Optional[Union[Optimizer, List[Optimizer]]] = None, # can be a single optimizer or List of optimizer
            finetune: bool = False
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer 

        assert len(metric_names) > 0 and len(loss_names) > 0
        self.metric_names = metric_names
        self.loss_names = loss_names
        self.ft_loss_names = ft_loss_names

        self.cur_epoch = 0
        self.cur_iter = 0
        
        self.finetune = finetune
    
        # Create the train and val (test) loaders.
        self.train_loader = get_train_dataloader(self.cfg, finetune)
        self.val_loader = get_val_dataloader(self.cfg, finetune)
        self.test_loader = get_test_dataloader(self.cfg)

        # create optimizer
        if self.optimizer is None:
            self.create_optimizer()

        self.reg_loss = np.array(0)

    def create_optimizer(self):
        self.optimizer = optim.construct_optimizer(self.model, self.cfg, finetune = self.finetune)

    def train(self):
        best_metric = self.cfg.TRAIN.BEST_METRIC_INITIAL

        for optimizer in self.optimizer:
            optimizer.zero_grad()
        
        for cur_epoch in range(self.cfg.SOLVER.START_EPOCH, self.cfg.SOLVER.MAX_EPOCH) if not self.finetune else range(self.cfg.SOLVER_FT.START_EPOCH, self.cfg.SOLVER_FT.MAX_EPOCH):
                                
            self.train_epoch()
            
            # Evaluate the model on validation set.
            if self._is_eval_epoch(cur_epoch):
                tracking_meter, loss_meter = self.eval_epoch()
                # check improvement
                is_best = self._check_improvement(tracking_meter.avg, best_metric) # validation mae average
                # Save a checkpoint on improvement.
                if is_best:
                    if not self.finetune:
                        with open(mkdir(self.cfg.RESULT_DIR) / "best_result.txt", 'w') as f:
                            f.write(f"Val/{tracking_meter.name}: {tracking_meter.avg} \
                                {loss_meter.name}: {loss_meter.avg}\tEpoch: {self.cur_epoch}")
                    else:
                        with open(mkdir(self.cfg.RESULT_DIR) / "best_result_ft.txt", 'w') as f:
                            f.write(f"Val/{tracking_meter.name}: {tracking_meter.avg} \
                                {loss_meter.name}: {loss_meter.avg}\tEpoch: {self.cur_epoch}")
                    self.save_best_model(tracking_meter, loss_meter)
                    best_metric = tracking_meter.avg
                    
            if hasattr(self.model, 'momentum') and self.model.momentum is not None:
                self.model.reset_momentum()
            
            self.cur_epoch += 1

    def _check_improvement(self, cur_metric, best_metric):
        if (self.cfg.TRAIN.BEST_LOWER and cur_metric < best_metric) \
                or (not self.cfg.TRAIN.BEST_LOWER and cur_metric > best_metric):
            return True
        else:
            return False

    def train_epoch(self):
        # set meters
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        if self.finetune:
            ft_loss_meters = self._get_ft_loss_meters()

            progress = ProgressMeter(
                len(self.train_loader),
                [batch_time, data_time, *metric_meters, *loss_meters, *ft_loss_meters],
                prefix="Epoch: [{}]".format(self.cur_epoch)
            )
        else:
            progress = ProgressMeter(
                len(self.train_loader),
                [batch_time, data_time, *metric_meters, *loss_meters],
                prefix="Epoch: [{}]".format(self.cur_epoch)
            )


        # switch to train mode
        self.model.train()

        data_size = len(self.train_loader)
        self.data_size = data_size

        start = time.time()
        fixed_batch = self.cfg.TRAIN.FINETUNE.BATCH_SIZE
        
        for cur_iter, inputs in enumerate(self.train_loader):
            self.cur_iter = cur_iter
            # dictionary for logging values
            log_dict = {}

            # measure data loading time
            data_time.update(time.time() - start)

            # Update the learning rate.
            lr = optim.get_epoch_lr(self.cur_epoch + (float(cur_iter)+0.5) / data_size, cur_iter, data_size, fixed_batch, self.cfg, self.finetune)
            #print('---',lr)
            optim.set_lr(self.optimizer, lr)

            # log to W&B #TODO 이거 lr 이제 list인데 이게 되나? 전반적인 wandb 수정해야 될 수도?
            log_dict.update({
                "lr/": lr
            })
            
            outputs = self.train_step(inputs)

            # update metric and loss meters, and log to W&B
            batch_size = self._find_batch_size(inputs)
            self._update_metric_meters(metric_meters, outputs["metrics"], batch_size)
            self._update_loss_meters(loss_meters, outputs["losses"], batch_size)
            log_dict.update({
                f"Train/{metric_meter.name}": metric_meter.val for metric_meter in metric_meters
            })
            log_dict.update({
                f"Train/{loss_meter.name}": loss_meter.val for loss_meter in loss_meters
            })
            if self.finetune:
                self._update_ft_loss_meters(ft_loss_meters, outputs["ft_losses"], batch_size)
                log_dict.update({
                    f"Train/{ft_loss_meter.name}": ft_loss_meter.val for ft_loss_meter in ft_loss_meters
                })

            if cur_iter % self.cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(cur_iter)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if self.cfg.WANDB.ENABLE:
                wandb.log(log_dict)

        log_dict = {}

        # log to W&B
        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict, commit=False)

    def _get_metric_meters(self):
        return [AverageMeter(metric_name, ":.4f") for metric_name in self.metric_names]

    def _get_loss_meters(self):
        return [AverageMeter(f"Loss {loss_name}", ":.4f") for loss_name in self.loss_names]

    def _get_ft_loss_meters(self):
        return [AverageMeter(f"FT_Loss {loss_name}", ":.4f") for loss_name in self.ft_loss_names]

    @staticmethod
    def _update_metric_meters(metric_meters, metrics, batch_size):
        assert len(metric_meters) == len(metrics)
        for metric_meter, metric in zip(metric_meters, metrics):
            metric_meter.update(metric.item(), batch_size)

    @staticmethod
    def _update_loss_meters(loss_meters, losses, batch_size):
        assert len(loss_meters) == len(losses)
        for loss_meter, loss in zip(loss_meters, losses):
            loss_meter.update(loss.item(), batch_size)

    @staticmethod
    def _update_ft_loss_meters(ft_loss_meters, ft_losses, batch_size):
        assert len(ft_loss_meters) == len(ft_losses)
        for ft_loss_meter, ft_loss in zip(ft_loss_meters, ft_losses):
            ft_loss_meter.update(ft_loss.item(), batch_size)
    
    def train_step(self, inputs): 
        # override for different methods
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        
        ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].float()
        dec_zeros = torch.zeros_like(dec_window[:, -self.cfg.DATA.PRED_LEN:, :]).float()
        dec_window = torch.cat([dec_window[:, :self.cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().cuda()
        
        # model_cfg = getattr(self.cfg.MODEL, self.cfg.MODEL_NAME.upper())
        model_cfg = self.cfg.MODEL
       
        if model_cfg.output_attention:
            pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp, finetune = self.finetune)[0]
        else:
            pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp, finetune = self.finetune)
        
        pred = pred[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:]
        
        #! config 넣어주는거랑 상관없이 이렇게 되니까 원하면 수정해라
        #loss = F.mse_loss(pred, ground_truth)
        #'''
        step = self.cur_iter*self.cfg.TRAIN.FINETUNE.BATCH_SIZE
        if self.finetune and step < self.cfg.SOLVER_FT.WARM_UP and self.cfg.TRAIN.FINETUNE.BATCH_SIZE>1:
            loss = F.mse_loss(pred, ground_truth, reduction='none')
            batch = loss.shape[0]
            loss = torch.reshape(loss, (batch, -1)).mean(1)
            #print('loss', loss.shape) #64, 96, 21

            loss_out = torch.mean(loss).clone().detach()
            
            adjust = torch.range(step+1, step+batch)/(self.cfg.SOLVER_FT.WARM_UP*batch)
            adjust[adjust > 1] = 1
            loss = torch.matmul(loss, adjust.to(loss.device))
            
        else:
            loss = F.mse_loss(pred, ground_truth) 
            loss_out = loss.clone().detach()
         
        #'''
        metric = F.l1_loss(pred, ground_truth)
        #loss_out = loss.clone().detach()

        if self.finetune:
            loss = loss / self.cfg.TRAIN.FINETUNE.ACCUM_SIZE
            if self.cfg.TRAIN.FINETUNE.REGULARIZATION == 'ewc' and (self.cur_iter+1) % self.cfg.TRAIN.FINETUNE.ACCUM_SIZE == 0:
                loss_reg = self.cfg.TRAIN.FINETUNE.REG_COEFF*self.reg_cls.penalty(self.model)
                self.reg_loss = loss_reg.clone().detach()
                loss += loss_reg
        loss.backward()

        if self.finetune and (self.cur_iter+1) % self.cfg.TRAIN.FINETUNE.ACCUM_SIZE != 0 and (self.cur_iter+1) != self.data_size:
            pass
        else:
            if self.finetune:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
            for optimizer in self.optimizer:
                if type(optimizer).__name__ == 'SAM':
                    raise ValueError('SAM not supported yet')
                    # optimizer.first_step(zero_grad=True)

                    # if model_cfg.output_attention:
                    #     pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp, finetune = self.finetune)[0]
                    # else:
                    #     pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp, finetune = self.finetune)

                    # loss = F.mse_loss(pred, ground_truth)
                    # metric = F.l1_loss(pred, ground_truth)

                    # loss.backward()
                    # optimizer.second_step(zero_grad=False)
                else:
                    optimizer.step()

                optimizer.zero_grad()  # Bong changed location
        
        outputs = dict(
            losses=(loss_out, ),
            ft_losses=(self.reg_loss, ),
            metrics=(metric,)
        )
        
        return outputs

    def _load_from_checkpoint(self):
        raise NotImplementedError

    def _find_batch_size(self, inputs):
        """
        Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
        """
        if isinstance(inputs, (list, tuple)):
            for t in inputs:
                result = self._find_batch_size(t)
                if result is not None:
                    return result
        elif isinstance(inputs, Mapping):
            for key, value in inputs.items():
                result = self._find_batch_size(value)
                if result is not None:
                    return result
        elif isinstance(inputs, torch.Tensor):
            return inputs.shape[0] if len(inputs.shape) >= 1 else None
        elif isinstance(inputs, np.ndarray):
            return inputs.shape[0] if len(inputs.shape) >= 1 else None

    def _is_eval_epoch(self, cur_epoch):
        return (cur_epoch + 1 == self.cfg.SOLVER.MAX_EPOCH) or (cur_epoch + 1) % self.cfg.TRAIN.EVAL_PERIOD == 0

    @torch.no_grad()
    def eval_epoch(self):
        # set meters
        batch_time = AverageMeter('Time', ':6.3f') 
        data_time = AverageMeter('Data', ':6.3f') 
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        progress = ProgressMeter(
            len(self.val_loader),
            [batch_time, data_time, *metric_meters, *loss_meters],
            prefix="Validation epoch[{}]".format(self.cur_epoch) 
        )
        log_dict = {}

        # switch to eval mode
        self.model.eval()

        start = time.time()
        mse_all_ = []
        mae_all_ = []
        for cur_iter, inputs in enumerate(self.val_loader):
            # measure data loading time
            data_time.update(time.time() - start)

            outputs = self.eval_step(inputs) # dictionary loss mse 값, metric mae 값 가지고 있음

            # update metric and loss meters, and log to W&B
            # batch_size = self._find_batch_size(inputs)
            # self._update_metric_meters(metric_meters, outputs["metrics"], batch_size)
            # self._update_loss_meters(loss_meters, outputs["losses"], batch_size)

            # if self._is_display_iter(cur_iter):
            #     progress.display(cur_iter)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()
            #'''
            mse_all_.append(outputs[0])
            mae_all_.append(outputs[1])
            #'''
        #'''
        mse_all_ = torch.flatten(torch.cat(mse_all_, dim=0))
        mae_all_ = torch.flatten(torch.cat(mae_all_, dim=0))
        
        
        if self.finetune and self.cfg.VAL.FINETUNE.CUT_FRONT>0:
            #print('valcut - front cut')
            mse_all_ = mse_all_[(self.cfg.DATA.PRED_LEN-1):]
            mae_all_ = mae_all_[(self.cfg.DATA.PRED_LEN-1):]
        if self.finetune and self.cfg.VAL.FINETUNE.CUT_FRONT>1:
            #print('valcut - adjust weight')
            total_len = len(mse_all_)
            sin_weight = 0.5*(torch.sin(torch.arange(total_len) * 0.5 * torch.Tensor([math.pi]) / total_len ))+0.5
            sin_weight = sin_weight.to(mse_all_.device)
            val_mse = torch.dot(mse_all_, sin_weight)/torch.sum(sin_weight)
            val_mae = torch.dot(mae_all_, sin_weight)/torch.sum(sin_weight)
        else:
            total_len = len(mse_all_)
            val_mse = mse_all_.mean()
            val_mae = mae_all_.mean()
        
        self._update_loss_meters(loss_meters, (val_mse,), total_len)
        self._update_metric_meters(metric_meters, (val_mae,), total_len)
        progress.display(cur_iter)
        #'''

        log_dict.update({
            f"Val/{metric_meter.name}": metric_meter.avg for metric_meter in metric_meters
        })
        log_dict.update({
            f"Val/{loss_meter.name}": loss_meter.avg for loss_meter in loss_meters
        })

        if self.cfg.WANDB.ENABLE:
            wandb.log(log_dict)

        # track the best model based on the first metric
        tracking_meter = metric_meters[0]

        return tracking_meter, loss_meters[0]

    @torch.no_grad()
    def eval_step(self, inputs):
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        
        ground_truth = dec_window[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:].float()
        dec_zeros = torch.zeros_like(dec_window[:, -self.cfg.DATA.PRED_LEN:, :]).float()
        dec_window = torch.cat([dec_window[:, :self.cfg.DATA.LABEL_LEN:, :], dec_zeros], dim=1).float().cuda()
        
        # model_cfg = getattr(self.cfg.MODEL, self.cfg.MODEL_NAME.upper())
        model_cfg = self.cfg.MODEL
        
        if model_cfg.output_attention:
            pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp, finetune = self.finetune)[0]
        else:
            pred = self.model(enc_window, enc_window_stamp, dec_window, dec_window_stamp, finetune = self.finetune)
        
        pred = pred[:, -self.cfg.DATA.PRED_LEN:, self.cfg.DATA.TARGET_START_IDX:]
        
        loss = F.mse_loss(pred, ground_truth, reduction='none')
        metric = F.l1_loss(pred, ground_truth, reduction='none')

        loss = torch.reshape(loss, (loss.size(0), -1)).mean(1)
        metric = torch.reshape(metric, (loss.size(0), -1)).mean(1)
        
        return (loss, metric)

        # outputs = dict(
        #     losses=(loss.mean(),),
        #     metrics=(metric.mean(),)
        # )
        #
        # return outputs

    def _is_display_iter(self, cur_iter):
        return cur_iter % self.cfg.TRAIN.PRINT_FREQ == 0 or (cur_iter + 1) == len(self.val_loader)

    @torch.no_grad()
    def predict(self):
        raise NotImplementedError #! 이거 안쓰이는거 같아서 이렇게 변경해둠
        self.load_best_model()

        # set to eval mode
        self.model.eval()

        # set meters
        metric_meters = self._get_metric_meters()
        loss_meters = self._get_loss_meters()
        progress = ProgressMeter(
            len(self.test_loader),
            [*metric_meters, *loss_meters],
            prefix="Test"
        )

        for cur_iter, inputs in enumerate(self.test_loader):
            inputs = prepare_inputs(inputs)
            outputs = self.model.get_anomaly_scores(inputs)

    def save_best_model(self, tracking_meter, loss_meter): 
        checkpoint = {
            "epoch": self.cur_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": [optimizer.state_dict() for optimizer in self.optimizer],
            "cfg": self.cfg.dump(),
            "validation_mae": tracking_meter.avg,
            "validation_mse": loss_meter.avg,
        }
        if hasattr(self.model, 'momentum') and self.model.momentum is not None:
            # checkpoint["momentum_matrix"] = self.model.momentum.momentum_matrix
            checkpoint["momentum_matrix"] = [m.momentum_matrix for m in self.model.momentum]
        if not self.finetune:
            with open(mkdir(self.cfg.TRAIN.CHECKPOINT_DIR) / 'checkpoint_best.pth', "wb") as f:
                torch.save(checkpoint, f)
        else:
            with open(mkdir(self.cfg.TRAIN.FINETUNE.CHECKPOINT_DIR) / 'checkpoint_best_ft.pth', "wb") as f:
                torch.save(checkpoint, f)

    def load_best_model(self):
        if not self.finetune:
            if self.cfg.TRAIN.ENABLE:
                model_path = os.path.join(self.cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
            else:
                model_path = os.path.join(self.cfg.TRAIN.FINETUNE.RESUME_DIR, "checkpoint_best.pth")
        else:
            model_path = os.path.join(self.cfg.TRAIN.FINETUNE.CHECKPOINT_DIR, "checkpoint_best_ft.pth")
        
        if os.path.isfile(model_path):
            if not (self.cfg.TRAIN.ENABLE or self.finetune):
                '''
                resume_config_path = os.path.join(self.cfg.TRAIN.FINETUNE.RESUME_DIR, "config.txt")
                resume_cfg = load_yacs_config(resume_config_path)

                if check_cfgs_same(resume_cfg, self.cfg):
                    pass
                else:
                    self.train()
                    return self.load_best_model()
                '''


            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")

            state_dict = checkpoint['model_state']

            # Load the model state dict
            if not self.finetune:
                state_dict = {k: v for k, v in state_dict.items() if 'momentum' not in k}
            msg = self.model.load_state_dict(state_dict, strict=False) #TODO changed this for debug
            if not self.finetune:
                for key in msg.missing_keys:
                    assert 'momentum' in key
            else:
                assert set(msg.missing_keys) == set()

            # Assign each momentum_matrix to the corresponding Momentum module
            #TODO change this part by bong
            if self.finetune:
                print('Load momentum matrix from checkpoint!')
                # self.model.momentum.momentum_matrix = checkpoint['momentum_matrix'].to(self.model.momentum.learnable_matrix.device)
                for m, momentum_matrix in zip(self.model.momentum, checkpoint['momentum_matrix']):
                    m.momentum_matrix = momentum_matrix
                    m.momentum_matrix = momentum_matrix.to(m.learnable_matrix.device)
            else:
                if hasattr(self.model, 'momentum') and self.model.momentum is not None:
                    self.model.reset_momentum()

            print(f"Loaded pre-trained model from {model_path}")

        else:
            print("=> no checkpoint found at '{}'".format(model_path))
            if not (self.cfg.TRAIN.ENABLE or self.finetune):
                self.train()
                return self.load_best_model()

        return self.model

def build_trainer(cfg, model, finetune = False):
    trainer = Trainer(cfg, model, cfg.MODEL.METRIC_NAMES, cfg.MODEL.LOSS_NAMES, cfg.MODEL.FT_LOSS_NAMES, finetune = finetune)
    return trainer

def prepare_inputs(inputs):
    # move data to the current GPU
    if isinstance(inputs, torch.Tensor):
        return inputs.float().cuda()
    elif isinstance(inputs, (tuple, list)):
        return type(inputs)(prepare_inputs(v) for v in inputs)
