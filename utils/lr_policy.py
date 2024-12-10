# adapted from: https://github.com/facebookresearch/SlowFast


import math


def get_lr_at_epoch(cfg, cur_epoch, cur_iter, fixed_batch, data_size, finetune=False):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    if not finetune:
        solver = cfg.SOLVER
    else:
        solver = cfg.SOLVER_FT
        
    lr_list = []
    
    for i in range(len(solver.OPTIMIZING_METHOD)):
        lr = get_lr_func(solver.LR_POLICY[i])(solver, cur_epoch, i)
        # Perform warm up.
        if cur_epoch < solver.WARMUP_EPOCHS[i]:
            lr_start = solver.WARMUP_START_LR[i]
            lr_end = get_lr_func(solver.LR_POLICY[i])(solver, solver.WARMUP_EPOCHS[i], i)
            alpha = (lr_end - lr_start) / solver.WARMUP_EPOCHS[i]
            lr = cur_epoch * alpha + lr_start
        lr_list.append(lr)
        
    return lr_list


def lr_func_cosine(solver, cur_epoch, i): #TODO 이거 맞게 작동하는지 봐야 하지만 cosine warmup 안쓰면 상관 없음
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    offset = solver.WARMUP_EPOCHS[i]
    assert solver.COSINE_END_LR[i] < solver.BASE_LR[i]
    return (
        solver.COSINE_END_LR[i]
        + (solver.BASE_LR[i] - solver.COSINE_END_LR[i])
        * (
            math.cos(
                math.pi * (cur_epoch - offset) / (solver.MAX_EPOCH - offset)
            )
            + 1.0
        )
        * 0.5
    )

def lr_func_decay(solver, cur_epoch, i): #TODO 이거 맞게 작동하는지 봐야 하지만 cosine warmup 안쓰면 상관 없음
    """
    Retrieve the learning rate to specified values at specified epoch with the
    cosine learning rate schedule. Details can be found in:
    Ilya Loshchilov, and  Frank Hutter
    SGDR: Stochastic Gradient Descent With Warm Restarts.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return solver.BASE_LR[i] * (solver.LR_DECAY_RATE[i]**(cur_epoch//solver.LR_DECAY_STEP[i]))

def get_lr_func(lr_policy):
    """
    Given the configs, retrieve the specified lr policy function.
    Args:
        lr_policy (string): the learning rate policy to use for the job.
    """
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]
