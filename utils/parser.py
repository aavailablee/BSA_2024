import argparse
import sys
import copy

from config import get_cfg_defaults
from utils.gsheet import write_gsheet

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test-time adaptation of unsupervised anomaly detector"
    )
    parser.add_argument( #
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default=None,
        type=str,
    )
    parser.add_argument( #
        "--SHEET_NAME",
        type=str,
        default=None
    )

    parser.add_argument("--DATA.NAME", type=str, default=None)

    parser.add_argument(
        "--TRAIN.FINETUNE.ENABLE",
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--SEED",
        type=int,
        default=None
    )
    parser.add_argument( #
        "--RUN_IDX",
        type=int,
        default=None
    )
    parser.add_argument("--MODEL_NAME", type=str, default=None)
    parser.add_argument("--MODEL.task_name", type=str, default=None)
    parser.add_argument("--DATA.PRED_LEN", type=int, default=None)
    #Train
    parser.add_argument("--SOLVER.MAX_EPOCH", type=int, default=None)
    parser.add_argument("--TRAIN.BATCH_SIZE", type=int, default=None)
    parser.add_argument("--SOLVER.OPTIMIZING_METHOD", nargs='+', type=str, default=None)
    parser.add_argument("--SOLVER.BASE_LR", nargs='+', type=float, default=None)
    parser.add_argument("--SOLVER.WEIGHT_DECAY", nargs='+', type=float, default=None)
    parser.add_argument( "--SOLVER.LR_POLICY", nargs='+', type=str, default=None )
    parser.add_argument( "--SOLVER.WARMUP_EPOCHS", nargs='+', type=int, default=None )
    parser.add_argument( "--SOLVER.LR_DECAY_STEP", nargs='+', type=float, default=None )
    parser.add_argument( "--SOLVER.LR_DECAY_RATE", nargs='+', type=float, default=None )
    #Finetune
    parser.add_argument("--SOLVER_FT.MAX_EPOCH", type=int, default=None)
    parser.add_argument("--SOLVER_FT.OPTIMIZING_METHOD", nargs='+', type=str, default=None)
    parser.add_argument("--SOLVER_FT.BASE_LR", nargs = '+', type = float, default = None )
    parser.add_argument("--SOLVER_FT.WEIGHT_DECAY", nargs='+', type=float, default=None)
    parser.add_argument("--SOLVER_FT.WARM_UP", type=int, default=None)
    parser.add_argument( "--SOLVER_FT.LR_POLICY", nargs='+', type=str, default=None )
    parser.add_argument( "--SOLVER_FT.WARMUP_EPOCHS", nargs='+', type=int, default=None ) # No use
    parser.add_argument( "--SOLVER_FT.LR_DECAY_STEP", nargs='+', type=float, default=None )
    parser.add_argument( "--SOLVER_FT.LR_DECAY_RATE", nargs='+', type=float, default=None )

    parser.add_argument("--MODEL.momentum_params", nargs='+', type=float, default=None)
    parser.add_argument("--TRAIN.FINETUNE.LEARN_MOMENTUM", type=str2bool, default=None)
    parser.add_argument("--TRAIN.FINETUNE.BPTT", type=str2bool, default=None)
    parser.add_argument("--TRAIN.FINETUNE.BATCH_SIZE", type=int, default=None)
    parser.add_argument("--TRAIN.FINETUNE.ACCUM_SIZE", type=int, default=None)
    parser.add_argument("--TRAIN.FINETUNE.REGULARIZATION", type=str, default=None )
    parser.add_argument("--TRAIN.FINETUNE.REG_COEFF", type=float, default=None )

    parser.add_argument("--TRAIN.FINETUNE.RESUME_DIR", nargs='+', type=str, default=None )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def load_config(args):
    # Setup cfg.
    cfg = get_cfg_defaults()
    args_dict = vars(args)
    # if not cfg.SHEET_NAME == 'debug': #! use this line for debudgging
    change_model_config(cfg, args_dict)

    switch_list = []
    for key, value in args_dict.items():
        if value is not None:
            switch_list.append(key)
            switch_list.append(value)
    print(switch_list)

    if len(switch_list) >0:
        cfg.merge_from_list(switch_list)
    cls_gsheet = write_gsheet(copy.deepcopy(cfg), args_dict)

    return cfg, cls_gsheet

def change_model_config(cfg, args_dict):
    if args_dict["MODEL_NAME"].startswith("iTransformer"):
        if args_dict["DATA.NAME"] == "weather":
            cfg.MODEL.e_layers      = 3
            cfg.MODEL.d_layers      = 1
            cfg.MODEL.factor        = 3
            cfg.MODEL.d_model       = 512
            cfg.MODEL.d_ff          = 512
        elif args_dict["DATA.NAME"] == "traffic":
            cfg.MODEL.e_layers      = 4
            cfg.MODEL.d_layers      = 1
            cfg.MODEL.factor        = 3
            cfg.MODEL.d_model       = 512
            cfg.MODEL.d_ff          = 512
        elif args_dict["DATA.NAME"] == "electricity":
            cfg.MODEL.e_layers      = 3
            cfg.MODEL.d_layers      = 1
            cfg.MODEL.factor        = 3
            cfg.MODEL.d_model       = 512
            cfg.MODEL.d_ff          = 512
        elif args_dict["DATA.NAME"] == "ETTh1":
            cfg.MODEL.e_layers      = 2
            cfg.MODEL.d_layers      = 1
            cfg.MODEL.factor        = 3
            cfg.MODEL.d_model       = 128
            cfg.MODEL.d_ff          = 128
        elif args_dict["DATA.NAME"] == "ETTh2":
            cfg.MODEL.e_layers      = 2
            cfg.MODEL.d_layers      = 1
            cfg.MODEL.factor        = 3
            cfg.MODEL.d_model       = 128
            cfg.MODEL.d_ff          = 128
        elif args_dict["DATA.NAME"] == "ETTm1":
            cfg.MODEL.e_layers      = 2
            cfg.MODEL.d_layers      = 1
            cfg.MODEL.factor        = 3
            cfg.MODEL.d_model       = 128
            cfg.MODEL.d_ff          = 128
        elif args_dict["DATA.NAME"] == "ETTm2":
            cfg.MODEL.e_layers      = 2
            cfg.MODEL.d_layers      = 1
            cfg.MODEL.factor        = 3
            cfg.MODEL.d_model       = 128
            cfg.MODEL.d_ff          = 128
        elif args_dict["DATA.NAME"] == "exchange_rate":
            cfg.MODEL.e_layers      = 2
            cfg.MODEL.d_layers      = 1
            cfg.MODEL.factor        = 3
            cfg.MODEL.d_model       = 128
            cfg.MODEL.d_ff          = 128
        elif args_dict["DATA.NAME"] == "solar":
            cfg.MODEL.e_layers      = 3
            cfg.MODEL.d_layers      = 1
            cfg.MODEL.factor        = 3
            cfg.MODEL.d_model       = 512
            cfg.MODEL.d_ff          = 512
        elif args_dict["DATA.NAME"] == "PEMS03":
            cfg.MODEL.e_layers      = 3
            cfg.MODEL.d_layers      = 1
            cfg.MODEL.factor        = 3
            cfg.MODEL.d_model       = 512
            cfg.MODEL.d_ff          = 512
        elif args_dict["DATA.NAME"] == "energydata_complete":
            cfg.MODEL.e_layers      = 3
            cfg.MODEL.d_layers      = 1
            cfg.MODEL.factor        = 3
            cfg.MODEL.d_model       = 512
            cfg.MODEL.d_ff          = 512
        else:
            raise NotImplementedError
        
    else:
        raise NotImplementedError

    ############## DATA RATIO #################
    if args_dict["DATA.NAME"] in ['ETTm1', 'ETTm2']:
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.6, 0.2
    elif args_dict["DATA.NAME"] in ['ETTh1', 'ETTh2']:
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.6, 0.2
    elif args_dict["DATA.NAME"] == 'electricity':
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
    elif args_dict["DATA.NAME"] == 'traffic':
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
    elif args_dict["DATA.NAME"] == 'weather':
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
    elif args_dict["DATA.NAME"] == 'exchange_rate':
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
    elif args_dict["DATA.NAME"] == 'illness':
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
    elif args_dict["DATA.NAME"] == 'solar':
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
    elif args_dict["DATA.NAME"] == 'PEMS03':
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
    elif args_dict["DATA.NAME"] == 'energydata_complete':
        cfg.DATA.TRAIN_RATIO, cfg.DATA.TEST_RATIO = 0.7, 0.2
    else:
        raise NotImplementedError
    
    return cfg