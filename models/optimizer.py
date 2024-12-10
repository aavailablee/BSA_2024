# reference: https://github.com/facebookresearch/SlowFast
import torch
import utils.lr_policy as lr_policy
from utils.sam import SAM

def get_optimizer(optim_params, cfg, finetune = False):
    if not finetune:
        solver = cfg.SOLVER
    else:
        solver = cfg.SOLVER_FT
    
    optimizers = []

    for i in range(len(solver.OPTIMIZING_METHOD)): 
        if solver.OPTIMIZING_METHOD[i] == "sgd":
            optimizer = torch.optim.SGD(
                optim_params[i],
                lr=solver.BASE_LR[i],
                momentum=solver.MOMENTUM[i],
                weight_decay=solver.WEIGHT_DECAY[i],
                dampening=solver.DAMPENING[i],
                nesterov=solver.NESTEROV[i],
            )
        elif solver.OPTIMIZING_METHOD[i] == 'adam':
            optimizer = torch.optim.Adam(
                optim_params[i],
                lr=solver.BASE_LR[i],
                weight_decay=solver.WEIGHT_DECAY[i]
            )
        elif solver.OPTIMIZING_METHOD[i] == 'Radam':
            optimizer = torch.optim.RAdam(
                optim_params[i],
                lr=solver.BASE_LR[i],
                weight_decay=solver.WEIGHT_DECAY[i]
            )
        elif solver.OPTIMIZING_METHOD[i] == 'adamW':
            optimizer = torch.optim.AdamW(
                optim_params[i],
                lr=solver.BASE_LR[i],
                weight_decay=solver.WEIGHT_DECAY[i]
            )
        elif solver.OPTIMIZING_METHOD[i] == 'sam_adam':
            base_optimizer = torch.optim.Adam 
            optimizer = SAM(
                optim_params[i], 
                base_optimizer, 
                lr=solver.BASE_LR[i], 
                weight_decay=solver.WEIGHT_DECAY[i],
                rho=0.05,
                adaptive=False
            )
        elif solver.OPTIMIZING_METHOD[i] == 'sam_Radam':
            base_optimizer = torch.optim.RAdam 
            optimizer = SAM(
                optim_params[i], 
                base_optimizer, 
                lr=solver.BASE_LR[i], 
                weight_decay=solver.WEIGHT_DECAY[i],
                rho=0.05,
                adaptive=False
            )
        elif solver.OPTIMIZING_METHOD[i] == 'sam_adamW':
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(
                optim_params[i], 
                base_optimizer, 
                lr=solver.BASE_LR[i], 
                weight_decay=solver.WEIGHT_DECAY[i],
                rho=0.05,
                adaptive=False
            )
        else:
            raise NotImplementedError(
                "Does not support {} optimizer".format(solver.OPTIMIZING_METHOD[i])
            )
        optimizers.append(optimizer)
        
    return optimizers


def construct_optimizer(model, cfg, finetune=False):
    optim_params = get_param_groups(model, cfg, finetune) 
    optimizer = get_optimizer(optim_params, cfg, finetune=finetune)

    return optimizer


def get_epoch_lr(cur_epoch, cur_iter, data_size, fixed_batch, cfg, finetune=False):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decay.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    if not finetune:
        solver = cfg.SOLVER
    else:
        solver = cfg.SOLVER_FT
        
    if solver.LR_POLICY is not None: #TODO cosine 쓸거면 제대로 된건지 확인해라 !Now error
        adj_lr = lr_policy.get_lr_at_epoch(cfg, cur_epoch, cur_iter, fixed_batch, data_size, finetune)
        if finetune and solver.WARM_UP:
            if cur_iter * fixed_batch < solver.WARM_UP:
                ratio_ = 0.5 * ((2 * cur_iter * fixed_batch + fixed_batch + 1) / solver.WARM_UP) - (0.5 * (solver.WARM_UP / fixed_batch)) * max(((cur_iter * fixed_batch + fixed_batch) / solver.WARM_UP) - 1, 0) ** 2
                return [val * ratio_ for val in adj_lr]
        return adj_lr

    elif finetune and solver.WARM_UP:
        if cur_iter*fixed_batch < solver.WARM_UP:
            ratio_ = 0.5*((2*cur_iter*fixed_batch+fixed_batch+1)/solver.WARM_UP)-(0.5*(solver.WARM_UP/fixed_batch))*max(((cur_iter*fixed_batch+fixed_batch)/solver.WARM_UP)-1,0)**2
            return [val*ratio_ for val in solver.BASE_LR]
        else:
            return solver.BASE_LR
    else:
        return solver.BASE_LR # List

def set_lr(optimizer, new_lr): # optimizer, new_lr 둘다 List
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for i in range(len(optimizer)):
        for idx, param_group in enumerate(optimizer[i].param_groups):
            param_group["lr"] = new_lr[i]
    # for idx, param_group in enumerate(optimizer.param_groups):
    #     param_group["lr"] = new_lr


def get_param_groups(model, cfg, finetune):
    if not finetune:
        param_groups = [[p for n, p in model.named_parameters() if p.requires_grad]]
    else:
        if cfg.TRAIN.FINETUNE.LEARN_MOMENTUM:
            param_groups = [[], [], []]
        else:
            param_groups = [[], []]
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(component in name for component in cfg.TRAIN.FINETUNE.COMPONENTS_TO_UPDATE_A):
                    param_groups[1].append(param)
                elif cfg.TRAIN.FINETUNE.LEARN_MOMENTUM and any(component in name for component in cfg.TRAIN.FINETUNE.COMPONENTS_TO_UPDATE_B):
                    param_groups[2].append(param)
                else:
                    param_groups[0].append(param)
    #   # Add all other parameters that requires_grad except the parameters in the param_group[0]
    #     for name, param in model.named_parameters(): 
    #         if id(param) not in map(id, param_groups[0]):
    #             param_groups[1].append(param)        

    return param_groups
