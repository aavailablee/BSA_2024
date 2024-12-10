import os
from datetime import datetime
from pytz import timezone

import wandb
from yacs.config import CfgNode as CN

from utils.misc import mkdir
import copy

def init_wandb(cfg):
    wandb.init(
        project=cfg.WANDB.PROJECT,
        name=cfg.WANDB.NAME,
        job_type=cfg.WANDB.JOB_TYPE,
        notes=cfg.WANDB.NOTES,
        dir=cfg.WANDB.DIR,
        config=cfg
    )
    # save checkpoints and results in the wandb log directory
    cfg.TRAIN.CHECKPOINT_DIR = str(mkdir(os.path.join(cfg.TRAIN.CHECKPOINT_DIR, wandb.run.dir)))
    cfg.TRAIN.FINETUNE.CHECKPOINT_DIR = str(mkdir(os.path.join(cfg.TRAIN.FINETUNE.CHECKPOINT_DIR, wandb.run.dir)))
    cfg.RESULT_DIR = str(mkdir(os.path.join(cfg.RESULT_DIR, wandb.run.dir)))
    

def set_time_to_log_dir(cfg: CN):

    cfg.RESULT_DIR = f'./results/{cfg.DATA.NAME}-{cfg.MODEL_NAME.rstrip("123456789")}-{cfg.MODEL.task_name}//'
    if not cfg.SHEET_NAME == 'debug':
        cfg.TRAIN.ENABLE = bool(not cfg.TRAIN.FINETUNE.ENABLE)
    cfg.TRAIN.CHECKPOINT_DIR = cfg.SHEET_NAME
    cfg.MODEL.batch_size = cfg.TRAIN.FINETUNE.BATCH_SIZE # Need for momentum module initialization
    cfg.MODEL.bptt = cfg.TRAIN.FINETUNE.BPTT  # Need for momentum module forward
    if cfg.TRAIN.FINETUNE.BATCH_SIZE>1: cfg.TRAIN.FINETUNE.ACCUM_SIZE = 1
    cfg.VAL.FINETUNE.BATCH_SIZE = cfg.TRAIN.FINETUNE.BATCH_SIZE
    cfg.MODEL.LEARN_MOMENTUM = cfg.TRAIN.FINETUNE.LEARN_MOMENTUM
    if not cfg.TRAIN.FINETUNE.LEARN_MOMENTUM: cfg.SOLVER_FT.OPTIMIZING_METHOD = cfg.SOLVER_FT.OPTIMIZING_METHOD[:2]
    cfg.VAL.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
    cfg.MODEL.pred_len = cfg.DATA.PRED_LEN

    ##### change setting ####
    # if cfg.SHEET_NAME.endswith('v1'):
    #     cfg.VAL.FINETUNE.CUT_FRONT = 1
    # elif cfg.SHEET_NAME.endswith('v2'):
    #     cfg.VAL.FINETUNE.CUT_FRONT = 2

    formatted_time = datetime.now(timezone('Asia/Seoul')).strftime("%y%m%d-%H%M%S.%f")[:-3]
    cfg.folder_code = formatted_time
    basic_path = copy.deepcopy(cfg.RESULT_DIR)
    adv_path = copy.deepcopy(cfg.TRAIN.CHECKPOINT_DIR)
    load_seed = copy.deepcopy(cfg.TRAIN.FINETUNE.RESUME_DIR[0])
    load_time = copy.deepcopy(cfg.TRAIN.FINETUNE.RESUME_DIR[1])
    if cfg.TRAIN.FINETUNE.ENABLE:
        cfg.RESULT_DIR = str(mkdir(os.path.join(basic_path, adv_path, f'finetune-{cfg.DATA.PRED_LEN}', cfg.MODEL_NAME, f'seed{cfg.SEED}', formatted_time)))
        cfg.TRAIN.FINETUNE.CHECKPOINT_DIR = cfg.RESULT_DIR
        cfg.TRAIN.FINETUNE.RESUME_DIR = str(mkdir(os.path.join(basic_path, adv_path, f'basic-{cfg.DATA.PRED_LEN}', f'seed{load_seed}', load_time)))
        cfg.TRAIN.CHECKPOINT_DIR = cfg.TRAIN.FINETUNE.RESUME_DIR
        cfg.write_dir = str(mkdir(os.path.join(basic_path, adv_path, f'finetune-{cfg.DATA.PRED_LEN}', cfg.MODEL_NAME))) #gsheet 62 line
    else:
        cfg.RESULT_DIR = str(mkdir(os.path.join(basic_path, adv_path, f'basic-{cfg.DATA.PRED_LEN}', f'seed{cfg.SEED}', formatted_time)))
        cfg.TRAIN.CHECKPOINT_DIR = cfg.RESULT_DIR
        ## No use
        cfg.TRAIN.FINETUNE.RESUME_DIR = cfg.RESULT_DIR
        cfg.TRAIN.FINETUNE.CHECKPOINT_DIR = cfg.RESULT_DIR
        ##
        cfg.write_dir = str(mkdir(os.path.join(basic_path, adv_path, f'basic-{cfg.DATA.PRED_LEN}')))
