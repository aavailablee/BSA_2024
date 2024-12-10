from yacs.config import CfgNode as CN

_C = CN()
# random seed number
_C.SEED = 0
# number of gpus per node. per node -> when using multiple servers
_C.NUM_GPUS = 1
_C.VISIBLE_DEVICES = 0
_C.LOG_TIME = True

_C.SHEET_NAME = 'debug' # use 'debug' for debugging #! check
_C.RUN_IDX = -1

_C.DATA_LOADER = CN()
# the number of data loading workers per gpu
_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.PIN_MEMORY = True
_C.DATA_LOADER.PREFETCH_FACTOR = 2
_C.DATA_LOADER.PERSISTENT_WORKERS = True

_C.DATA = CN()
_C.DATA.BASE_DIR = './data/'
_C.DATA.NAME = 'weather' #! check traffic
_C.DATA.N_VAR = 21 #! check
_C.DATA.SEQ_LEN = 96 # encoder input window length (look back window)
_C.DATA.LABEL_LEN = 48 # length entering decoder embedding (half of seq_len here) (not needed in itransformer)
_C.DATA.PRED_LEN = 96 # prediction length (mainly change this)
_C.DATA.FEATURES = 'M' # prediction target is multivariate, 'S' means input: multivariate, prediction: univariate
_C.DATA.TIMEENC = 0 # depending on how the time stamp is in the data, 0 or 1
_C.DATA.FREQ = 'h' # h or t. temporal embedding. t includes minute embedding. h includes hour, weekday, day, month embedding (not needed in iTransformer)
_C.DATA.SCALE = "standard" # initial preprocessing normalization method # standard, min-max, none
_C.DATA.TRAIN_RATIO = 0.7 # train, val, test ratio, data is split in the order of train, val, test
_C.DATA.TEST_RATIO = 0.15 # train, val, test ratio
_C.DATA.DATE_IDX = 0 # column index to drop date from raw data #! check
_C.DATA.TARGET_START_IDX = 0 # column index where prediction target starts after dropping columns (prediction variables should be at the end)

_C.TRAIN = CN()
_C.TRAIN.ENABLE = False # whether to train in main.py #! check
_C.TRAIN.SPLIT = 'train'
_C.TRAIN.BATCH_SIZE = 64 #32 #! check
_C.TRAIN.SHUFFLE = True
_C.TRAIN.DROP_LAST = True  # whether to drop the last batch if the dataset length is not divisible by batch_size
# _C.TRAIN.RESUME = '' # path to checkpoint to resume training
# _C.TRAIN.CHECKPOINT_PERIOD = 200 # epoch period to save checkpoints
_C.TRAIN.EVAL_PERIOD = 1 # epoch period to evaluate on a validation set
_C.TRAIN.PRINT_FREQ  = 1 # iteration frequency to print progress meter
_C.TRAIN.BEST_METRIC_INITIAL = float("inf") # initial value for best model tracking (MSE or MAE)
_C.TRAIN.BEST_LOWER = True # whether lower metric is better
_C.TRAIN.COMPONENTS_TO_DISABLE = ['momentum'] # components to disable #! check

_C.VAL = CN()
_C.VAL.SPLIT = 'val'
_C.VAL.BATCH_SIZE = _C.TRAIN.BATCH_SIZE #! check
_C.VAL.SHUFFLE = False
_C.VAL.DROP_LAST = False
_C.VAL.VIS = False

_C.TEST = CN()
_C.TEST.ENABLE = True # whether to test in main.py
_C.TEST.SPLIT = 'test'
_C.TEST.BATCH_SIZE = 1 #! check
_C.TEST.SHUFFLE = False
_C.TEST.DROP_LAST = False
_C.TEST.VIS_ERROR = True # whether to show error
_C.TEST.VIS_DATA = False # whether to show TOP, WORST data
_C.TEST.VIS_DATA_NUM = 5 # number of TOP, WORST data to show
_C.TEST.PREDICTION_ERROR_DIR = "" # directory to load prediction error, usually set to ""
_C.TEST.PREDICTION_ERROR_TYPE = "MAE" # MAE, MSE

_C.TEST.APPLY_MOVING_AVERAGE = False # apply moving average if True, for observing changes over time (related to TTA check)
_C.TEST.MOVING_AVERAGE_WINDOW = 100

_C.MODEL_NAME = 'DLinear_momentum' #! check 'iTransformer_momentum4'
_C.MODEL = CN()
_C.MODEL.task_name = 'long_term_forecast'
_C.MODEL.seq_len = _C.DATA.SEQ_LEN 
_C.MODEL.label_len = _C.DATA.LABEL_LEN # not needed in iTransformer
_C.MODEL.pred_len = _C.DATA.PRED_LEN 
_C.MODEL.e_layers = 2
_C.MODEL.d_layers = 1 # not needed in iTransformer
_C.MODEL.factor = 1 # was 3, used in Prob Attention (probabilistic attention) in informer
_C.MODEL.num_kernels = 6 # for Inception

_C.MODEL.enc_in = _C.DATA.N_VAR 
_C.MODEL.dec_in = _C.DATA.N_VAR # not needed in iTransformer
_C.MODEL.c_out = _C.DATA.N_VAR # not needed in iTransformer

_C.MODEL.d_model = 512 # embedding dimension
_C.MODEL.d_ff = 2048 # feedforward dimension d_model -> d_ff -> d_model

_C.MODEL.moving_avg = 25 # window size of moving average, seems to be used in autoformer

_C.MODEL.output_attention = False # whether the attention weights are returned by the forward method of the attention class
_C.MODEL.dropout = 0.1 #0.1
_C.MODEL.n_heads = 8
_C.MODEL.activation = 'gelu'
_C.MODEL.METRIC_NAMES = ('MAE',)
_C.MODEL.LOSS_NAMES = ('MSE',)
_C.MODEL.FT_LOSS_NAMES = ('REG',)

# positional embedding is the position within the window
# temporal embedding is the information about time
_C.MODEL.embed = 'timeF' # not needed in iTransformer
_C.MODEL.freq = 'h' # not needed in iTransformer 

_C.MODEL.TRAIN_COMPONENTS_TO_DISABLE = _C.TRAIN.COMPONENTS_TO_DISABLE

_C.MODEL.momentum_params = [0.9, 0.99, 0.999] #TODO momentum parameter

_C.SOLVER = CN()
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.MAX_EPOCH = 10
_C.SOLVER.OPTIMIZING_METHOD = ['adamW']
_C.SOLVER.BASE_LR = [0.001] # warmup end learning rate
_C.SOLVER.WEIGHT_DECAY = [0.01] #[0.01] 

_C.SOLVER.LR_POLICY = ['cosine'] # if removed, it returns to base_lr. for warmup, set this to cosine #cosine / decay
_C.SOLVER.COSINE_END_LR = [0.0]

_C.SOLVER.WARMUP_EPOCHS = [0.2] # linear warmup epoch
_C.SOLVER.WARMUP_START_LR = [0] # warmup start learning rate

_C.SOLVER.LR_DECAY_STEP = [1]
_C.SOLVER.LR_DECAY_RATE = [0.9]

# directory to save result txt file
_C.RESULT_DIR = None
_C.TRAIN.CHECKPOINT_DIR = None # directory to save checkpoints

_C.TRAIN.FINETUNE = CN()
_C.TRAIN.FINETUNE.ENABLE = True # whether to finetune in main.py #! check
_C.TRAIN.FINETUNE.RESUME_DIR = [0,f'00000-000000'] # checkpoint to resume training #! check (seed / time)
_C.TRAIN.FINETUNE.SPLIT = 'train'
_C.TRAIN.FINETUNE.BATCH_SIZE = 1 #! check
_C.TRAIN.FINETUNE.ACCUM_SIZE = 64 #! This is loss accumulation emulating batched training
_C.TRAIN.FINETUNE.BPTT = True
_C.TRAIN.FINETUNE.LEARN_MOMENTUM = False
_C.TRAIN.FINETUNE.REGULARIZATION = None#'ewc'
_C.TRAIN.FINETUNE.REG_COEFF = 1.
_C.TRAIN.FINETUNE.SHUFFLE = False
_C.TRAIN.FINETUNE.DROP_LAST = False  # whether to drop the last batch if the dataset length is not divisible by batch_size
_C.TRAIN.FINETUNE.CHECKPOINT_DIR = None#'./results/' # directory to save checkpoints
_C.TRAIN.FINETUNE.EVAL_PERIOD = 1 # epoch period to evaluate on a validation set
_C.TRAIN.FINETUNE.PRINT_FREQ  = 100 # iteration frequency to print progress meter

_C.TRAIN.FINETUNE.COMPONENTS_TO_UPDATE_A = ['learnable_matrix'] # components to update parameters #! check
_C.TRAIN.FINETUNE.COMPONENTS_TO_UPDATE_B = ['momentum_params_learnable'] # components to update parameters #! check
_C.TRAIN.FINETUNE.COMPONENTS_TO_DISABLE = [] # components to disable #! check
_C.MODEL.FINETUNE_COMPONENTS_TO_DISABLE = _C.TRAIN.FINETUNE.COMPONENTS_TO_DISABLE

_C.SOLVER_FT = CN()
_C.SOLVER_FT.START_EPOCH = 0
_C.SOLVER_FT.MAX_EPOCH = 1
_C.SOLVER_FT.OPTIMIZING_METHOD = ['adamW', 'adamW', 'adamW']
_C.SOLVER_FT.BASE_LR = [0.01 * 0.001, 0.01, 0.01 * 0.001, ] # warmup end learning rate
_C.SOLVER_FT.WEIGHT_DECAY = [0.01 * 0.01, 0.0, 0.0]
_C.SOLVER_FT.WARM_UP = 1000
# parameters for sgd only
# _C.SOLVER_FT.MOMENTUM = [0, 0]
# _C.SOLVER_FT.DAMPENING = [0.0, 0.0] # reduce the momentum's effect at the end of training. worth considering
# _C.SOLVER_FT.NESTEROV = [True, True]

_C.SOLVER_FT.LR_POLICY = None # if removed, it returns to base_lr. for warmup, set this to cosine
_C.SOLVER_FT.COSINE_END_LR = [0.0, 0.0, 0.0]

_C.SOLVER_FT.WARMUP_EPOCHS = [0, 0, 0] # linear warmup epoch
_C.SOLVER_FT.WARMUP_START_LR = [0, 0, 0] # warmup start learning rate

_C.SOLVER_FT.LR_DECAY_STEP = [1, 1, 1]
_C.SOLVER_FT.LR_DECAY_RATE = [0.9, 0.9, 0.9]

_C.VAL.FINETUNE = CN()
_C.VAL.FINETUNE.SPLIT = 'val'
_C.VAL.FINETUNE.BATCH_SIZE = 1
_C.VAL.FINETUNE.SHUFFLE = False
_C.VAL.FINETUNE.DROP_LAST = False
_C.VAL.FINETUNE.CUT_FRONT = 2

#! Currently, our momentum code uses multiple optimizers, but wandb implementation is not yet modified to support it
_C.WANDB = CN()
_C.WANDB.ENABLE = False # wandb on/off #! check
_C.WANDB.PROJECT = 'Momentum'
_C.WANDB.NAME = '' #! check
_C.WANDB.JOB_TYPE = 'train' # train or eval
_C.WANDB.NOTES = '' # a description of this run
_C.WANDB.DIR = './'
_C.WANDB.VIS_TRAIN_SCORE = False
_C.WANDB.VIS_TEST_SCORE = False 
_C.WANDB.VIS_TEST_DATA = False # visualize raw data
_C.WANDB.VIS_TRAIN_TEST_HISTOGRAM = False

def get_cfg_defaults():
    return _C.clone()
