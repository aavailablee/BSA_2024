import copy
import os
import sys
import argparse
import pickle
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class tk:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.counter = 0
    def run_(self,do, load=False):
        self.counter +=1
        '''
        if self.counter < self.start:
            return
        elif self.counter >= self.end:
            raise 'End'
        '''
        print('')
        print(f'{self.counter} - '+do)
        if self.set == 'finetune':
            read_path = f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/{self.set}-{self.len_pred}/{self.model_name}/tmp{self.counter}.pkl'
        else:
            read_path = f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/{self.set}-{self.len_pred}/tmp{self.counter}.pkl'

        if args.start <= self.counter:
            os.system(f'{do} --RUN_IDX {self.counter} --SHEET_NAME {self.sheet_name}') #main RUN

        if self.set == 'finetune':
            with open(read_path, "rb") as fr:
                data, folder_code =pickle.load(fr)
            with open(f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/{self.set}-{self.len_pred}/{self.model_name}/tmp{self.counter}_data.pkl', "wb") as fw:
                pickle.dump({'ft_lr_0':self.ft_lr_0, 'ft_lr_1':self.ft_lr_1, 'ft_lr_2':self.ft_lr_2, 'momentum_params':self.momentum_params},fw)
            if os.path.isfile(f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/{self.set}-{self.len_pred}/{self.model_name}/tmp{self.counter-1}_data.pkl'):
                os.remove(f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/{self.set}-{self.len_pred}/{self.model_name}/tmp{self.counter-1}_data.pkl')
        else:
            with open(read_path, "rb") as fr:
                data, folder_code = pickle.load(fr)
            with open(
                    f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/{self.set}-{self.len_pred}/tmp{self.counter}_data.pkl',
                    "wb") as fw:
                pickle.dump({'basic_lr': self.basic_lr, 'weight_decay': self.weight_decay, 'lr_policy': self.lr_policy,}, fw)
            if os.path.isfile(
                    f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/{self.set}-{self.len_pred}/tmp{self.counter - 1}_data.pkl'):
                os.remove(
                    f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/{self.set}-{self.len_pred}/tmp{self.counter - 1}_data.pkl')

        return data, folder_code

    def write_best_dir(self, best_dir, value=None):
        if self.set == 'basic':
            with open(
                    f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/{self.set}-{self.len_pred}/best_model_dir.pkl',
                    "wb") as fw:
                pickle.dump([[0, best_dir[0]], [1, best_dir[1]], [2, best_dir[2]]], fw)

            with open(
                    f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/{self.set}-{self.len_pred}/a_final_result.txt', "w") as f:
                f.write(f"{str([[0, best_dir[0]], [1, best_dir[1]], [2, best_dir[2]]])} - avg value: {str(np.mean(value, axis=0).tolist())} - std value: {str(np.std(value, axis=0).tolist())} - raw value: {str(value.tolist())} \n"
                        f"basic_lr    , {task.basic_lr}\n"
                        f"weight_decay, {task.weight_decay}\n"
                        f"lr_policy   , {str(task.lr_policy)}\n")

        else:
            with open(f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/{self.set}-{self.len_pred}/{self.model_name}/a_final_result.txt', "w") as f:
                f.write(f"{str([[0, best_dir[0]], [1, best_dir[1]], [2, best_dir[2]]])} - avg value: {str(np.mean(value, axis=0).tolist())} - std value: {str(np.std(value, axis=0).tolist())} - raw value: {str(value.tolist())} \n"
                        f"ft_lr_0 {task.ft_lr_0}\n"
                        f"ft_lr_1 {task.ft_lr_1}\n"
                        f"ft_lr_2 {task.ft_lr_2}\n"
                        f"momentum_params {str(task.momentum_params)}\n")


    def load_best(self):
        with open(f'./results/{self.data_name}-{self.model_name.rstrip("123456789")}-{self.task_name}/{self.sheet_name}/basic-{self.len_pred}/best_model_dir.pkl', "rb") as fr:
            return pickle.load(fr)


parser = argparse.ArgumentParser(description='use argparse to run main')  # 2. parser? ???.

# 3. parser.add_argument? ???? ??? ??????.
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--basic', default=None, action='store_true',)
parser.add_argument('--len', nargs='+', default=None, type=int)  # 0,1,2,3 --> 96/192/336/720
parser.add_argument('--data', nargs='+', default=None, type=int) # 0,1,... --> weather/traffic/...
parser.add_argument('--model', nargs='+',default=None, type=int) # 0,1,2... only for FT
parser.add_argument('--start', default=1, type=int)
parser.add_argument('--end', default=10000000, type=int)
args = parser.parse_args()  # 4. ??? ??
len_list = [96, 192, 336, 720]
for len_pred in args.len:
    assert len_pred in [0,1,2,3]
# static setting

cuda = args.cuda
optimizer = 'adam'
sheet_name = f'itransformer_{optimizer}' #! chcek


task            = tk(args.start, args.end)
task.sheet_name = sheet_name
task_name       = 'long_term_forecast'
model_name      = 'iTransformer_momentum1' #! Check model to use for pretraining
data_names      = ['weather', 'traffic', 'electricity', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'solar', 'PEMS03'] #! Check dataset names
max_epochs      = [40, 20, 30, 30, 30, 30, 30, 30, 30, 30] #! Check pretrain epochs for each dataset
basic_batchs    = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64] #! Check pretrain batch size for each dataset
ft_max_epochs   = [20, 20, 20, 30, 30, 20, 20, 30, 20, 20]
ft_batchs       = [256, 64, 256, 256, 256, 256, 256, 256, 256, 256] #! Check finetune batch size for each dataset

basics = {}
model_names = ['iTransformer_momentum1'] #! Check

def run_basic():
    ##### Freeze ######
    len_counter = 1000000
    data_counter = 10000
    task.set = 'basic'
    seeds = [0, 1, 2] #3 #! Check
    basic_lrs = [0.0003] #5
    basic_lr_init = 0.0003
    task.basic_lr = basic_lr_init
    weight_decays = [0.00003]
    weight_decay_init = 0.00003
    task.weight_decay = weight_decay_init
    lr_policys = [['decay', 0.9]] #, ['cosine',None]]
    lr_policy_init = ['decay', 0.9]
    task.lr_policy = lr_policy_init
    ###################

    def basic_search():
        tmp = []
        folder_codes = []
        for seed in seeds:
            task.model_name = model_name
            task.task_name = task_name
            task.data_name = data_name
            task.len_pred = len_pred
            out_list, folder_code = task.run_(f"CUDA_VISIBLE_DEVICES={cuda} python main.py --SEED {seed} --no-TRAIN.FINETUNE.ENABLE "
                      f"--SOLVER.OPTIMIZING_METHOD {optimizer} "
                      f"--MODEL_NAME {model_name} "
                      f"--MODEL.task_name {task_name} "
                      f"--DATA.PRED_LEN {len_pred} "
                      f"--DATA.NAME {data_name} "
                      f"--SOLVER.MAX_EPOCH {max_epoch} "
                      f"--TRAIN.BATCH_SIZE {basic_batch} "
                      f"--SOLVER.BASE_LR {basic_lr} "
                      f"--SOLVER.WEIGHT_DECAY {weight_decay} "
                      f"--SOLVER.LR_POLICY {lr_policy[0]} "
                      f"{f'--SOLVER.LR_DECAY_RATE {lr_policy[1]} ' if lr_policy[0] == 'decay' else ''}"
                      )
            tmp.append(out_list)
            folder_codes.append(folder_code)
        return tmp, folder_codes

    #2*3*6*(4+1) = 180
    for len_idx in args.len:
        task.counter = len_counter * (len_idx + 1)
        len_pred = len_list[len_idx]
        for data_idx in args.data:
            task.counter = (task.counter // len_counter) * len_counter
            task.counter += data_counter*(data_idx+1)
            data_name = data_names[data_idx]
            max_epoch = max_epochs[data_idx]
            basic_batch = basic_batchs[data_idx]
            
            ###init
            basic_lr = copy.deepcopy(basic_lr_init)
            weight_decay = copy.deepcopy(weight_decay_init)
            lr_policy = copy.deepcopy(lr_policy_init)

            out = []
            dir_ = []
            for basic_lr in basic_lrs:
                task.basic_lr = basic_lr
                r1, r2 = basic_search()
                out.append(r1)
                dir_.append(r2)
            out = np.array(out)  # len, 3, 6
            idx_basic_lr = np.argmin(np.mean(out[:, :, -2], 1)).item()
            out_prev = out[idx_basic_lr, :, :]
            best_dir = dir_[idx_basic_lr]
            basic_lr = basic_lrs[idx_basic_lr]
            task.basic_lr = basic_lr

            out = []
            dir_ = []
            prev_weight_decay = copy.deepcopy(weight_decay)
            for weight_decay in weight_decays:
                task.weight_decay = weight_decay
                if weight_decay == prev_weight_decay:
                    r1, r2 = out_prev, best_dir
                else:
                    r1, r2 = basic_search()
                out.append(r1)
                dir_.append(r2)
            out = np.array(out)  # len, 3, 6
            idx_weight_decay = np.argmin(np.mean(out[:, :, -2], 1)).item()
            out_prev = out[idx_weight_decay, :, :]
            best_dir = dir_[idx_weight_decay]
            weight_decay = weight_decays[idx_weight_decay]
            task.weight_decay = weight_decay

            out = []
            dir_ = []
            for ratio in [0.7, 1., 1.4]:
                basic_lr = ratio * basic_lrs[idx_basic_lr]
                task.basic_lr = basic_lr
                if ratio == 1.:
                    r1, r2 = out_prev, best_dir
                else:
                    r1, r2 = basic_search()
                out.append(r1)
                dir_.append(r2)
            out = np.array(out)  # len, 3, 6
            idx = np.argmin(np.mean(out[:, :, -2], 1)).item()
            out_prev = out[idx, :, :]
            best_dir = dir_[idx]
            basic_lr = [0.7, 1., 1.4][idx] * basic_lrs[idx_basic_lr]
            task.basic_lr = basic_lr

            out = []
            dir_ = []
            for ratio in [0.7, 1., 1.4]:
                weight_decay = ratio * weight_decays[idx_weight_decay]
                task.weight_decay = weight_decay
                if ratio == 1.:
                    r1, r2 = out_prev, best_dir
                else:
                    r1, r2 = basic_search()
                out.append(r1)
                dir_.append(r2)
            out = np.array(out)  # len, 3, 6
            idx = np.argmin(np.mean(out[:, :, -2], 1)).item()
            out_prev = out[idx, :, :]
            best_dir = dir_[idx]
            task.weight_decay = weight_decay
            task.write_best_dir(best_dir, out[idx])
            args.start = 1



def run_ft():
    ######### Freeze ##########
    len_counter = 1000000
    data_counter = 10000
    model_counter = 1000
    task.set = 'finetune'
    #########################

    def search(get_folder_code = False):
        if ft_lr_2 == None:
            ft_lr_2_tmp = 0
            learn_momentum = 'n'
        else:
            ft_lr_2_tmp = ft_lr_2
            learn_momentum = 'y'
        tmp = []
        folder_codes = []
        for basic in load_pretrained:
            task.model_name = model_name
            task.task_name = task_name
            task.data_name = data_name
            task.len_pred = len_pred
            string_ = ""
            for i in momentum_params: string_ += (str(i) + ' ')
            out_list, folder_code = task.run_(
                f"CUDA_VISIBLE_DEVICES={cuda} python main.py --SEED {basic[0]} "
                f"--MODEL_NAME {model_name} "
                f"--MODEL.task_name {task_name} "
                f"--DATA.PRED_LEN {len_pred} "
                f"--DATA.NAME {data_name} "
                f"--SOLVER_FT.OPTIMIZING_METHOD {optimizer} {optimizer} {optimizer} "
                f"--TRAIN.FINETUNE.RESUME_DIR {basic[0]} {basic[1]} "
                f"--SOLVER_FT.MAX_EPOCH {max_epoch} "
                f"--SOLVER_FT.BASE_LR {ft_lr_0} {ft_lr_1} {ft_lr_2_tmp} "
                f"--SOLVER_FT.LR_POLICY {lr_policy} {lr_policy} {lr_policy} "
                f"{f'--SOLVER_FT.LR_DECAY_RATE {decay_rate} {decay_rate} {decay_rate} ' if lr_policy == 'decay' else ''}"
                f"--TRAIN.FINETUNE.BATCH_SIZE {ft_batch} "
                f"--TRAIN.FINETUNE.ACCUM_SIZE {1} "
                f"--TRAIN.FINETUNE.BPTT {use_bptt} "
                f"--TRAIN.FINETUNE.LEARN_MOMENTUM {learn_momentum} "
                f"--MODEL.momentum_params {string_} "
                f"{f'--TRAIN.FINETUNE.REGULARIZATION ewc --TRAIN.FINETUNE.REG_COEFF {ft_reg_coeff} ' if ft_reg_coeff > 0 else ''}")
            tmp.append(out_list)
            folder_codes.append(folder_code)
        if get_folder_code:
            return tmp, folder_codes
        return tmp

    ft_rest_lrs = [0.00041999999999999996] # sensitive
    init_ft_lr_0 = 0.00041999999999999996
    task.ft_lr_0 = init_ft_lr_0
    ft_learnable_matrix_lrs = [0.020999999999999998]  # sensitive
    init_ft_lr_1 = 0.020999999999999998
    task.ft_lr_1 = init_ft_lr_1
    ft_momentum_params_learnable_lrs = [0.01] #less sensitive
    init_ft_lr_2 = 0.001
    task.ft_lr_2 = init_ft_lr_2
    momentum_params_s = [[0.9, 0.99, 0.999, 0.9999]] # Can use any length
    momentum_params_init = [0.9, 0.99, 0.999, 0.9999]
    task.momentum_params = momentum_params_init


    use_bptts = ['y']
    use_bptt = 'y'
    lr_policys = ['cosine'] #['decay', 'cosine']
    lr_policy = 'cosine'
    decay_rates = [None] #+1
    decay_rate = None
    #learn_momentum = ['y', 'n']
    ft_reg_coeffs = [0, ]  # 1  / 0, 100, 1000, 10000
    ft_reg_coeff = 0

    for len_pred_idx in args.len:
        task.counter = len_counter * (len_pred_idx + 1)
        len_pred = len_list[len_pred_idx]
        for data_idx in args.data:
            task.counter = (task.counter // (len_counter)) * (len_counter)
            task.counter += data_counter*(data_idx+1)
            data_name = data_names[data_idx]
            ft_batch = ft_batchs[data_idx]
            max_epoch = ft_max_epochs[data_idx]

            for model_idx in args.model:
                task.counter = (task.counter // (data_counter)) * (data_counter)
                task.counter += model_counter * (model_idx + 1)
                model_name = model_names[model_idx]
                
                ## init
                ft_lr_0 = copy.deepcopy(init_ft_lr_0)
                ft_lr_1 = copy.deepcopy(init_ft_lr_1)
                ft_lr_2 = copy.deepcopy(init_ft_lr_2)
                momentum_params = copy.deepcopy(momentum_params_init)

                if (len_pred in basics.keys()) and (data_name in basics[len_pred].keys()):
                    load_pretrained = basics[data_name]
                else:
                    task.data_name = data_name
                    task.model_name = model_name
                    task.task_name = task_name
                    task.len_pred = len_pred
                    load_pretrained = task.load_best()

                out = []
                for ft_lr_0 in ft_rest_lrs:
                    task.ft_lr_0 = ft_lr_0
                    out.append(search())
                out = np.array(out) #len, 3, 6
                idx_ft_lr_0 = np.argmin(np.mean(out[:,:,-2],1)).item()
                out_prev = out[idx_ft_lr_0,:,:]
                ft_lr_0 = ft_rest_lrs[idx_ft_lr_0]
                task.ft_lr_0 = ft_lr_0

                out = []
                prev_ft_lr_1 = copy.deepcopy(ft_lr_1)
                for ft_lr_1 in ft_learnable_matrix_lrs:
                    task.ft_lr_1 = ft_lr_1
                    if ft_lr_1 == prev_ft_lr_1:
                        out.append(out_prev)
                    else:
                        out.append(search())
                out = np.array(out)  # len, 3, 6
                idx_ft_lr_1 = np.argmin(np.mean(out[:, :, -2], 1)).item()
                out_prev = out[idx_ft_lr_1, :, :]
                ft_lr_1 = ft_learnable_matrix_lrs[idx_ft_lr_1]
                task.ft_lr_1 = ft_lr_1

                out = []
                prev_ft_lr_2 = copy.deepcopy(ft_lr_2)
                for ft_lr_2 in ft_momentum_params_learnable_lrs:
                    task.ft_lr_2 = ft_lr_2
                    if ft_lr_2 == prev_ft_lr_2:
                        out.append(out_prev)
                    else:
                        out.append(search())
                out = np.array(out)  # len, 3, 6
                idx_ft_lr_2 = np.argmin(np.mean(out[:, :, -2], 1)).item()
                out_prev = out[idx_ft_lr_2, :, :]
                ft_lr_2 = ft_momentum_params_learnable_lrs[idx_ft_lr_2]
                task.ft_lr_2 = ft_lr_2

                out = []
                prev_momentum_params = copy.deepcopy(momentum_params)
                for momentum_params in momentum_params_s:
                    task.momentum_params = momentum_params
                    if momentum_params == prev_momentum_params:
                        out.append(out_prev)
                    else:
                        out.append(search())
                out = np.array(out)  # len, 3, 6
                idx_momentum_params = np.argmin(np.mean(out[:, :, -2], 1)).item()
                out_prev = out[idx_momentum_params, :, :]
                momentum_params = momentum_params_s[idx_momentum_params]
                task.momentum_params = momentum_params

                out = []
                for ratio in [0.7,1.,1.4]:
                    ft_lr_0 = ratio*ft_rest_lrs[idx_ft_lr_0]
                    task.ft_lr_0 = ft_lr_0
                    if ratio == 1.:
                        out.append(out_prev)
                    else:
                        out.append(search())
                out = np.array(out) #len, 3, 6
                idx = np.argmin(np.mean(out[:,:,-2],1)).item()
                out_prev = out[idx, :, :]
                ft_lr_0 = [0.7,1.,1.4][idx]*ft_rest_lrs[idx_ft_lr_0]
                task.ft_lr_0 = ft_lr_0

                out = []
                for ratio in [0.7,1.,1.4]:
                    ft_lr_1 = ratio * ft_learnable_matrix_lrs[idx_ft_lr_1]
                    task.ft_lr_1 = ft_lr_1
                    if ratio == 1.:
                        out.append(out_prev)
                    else:
                        out.append(search())
                out = np.array(out)  # len, 3, 6
                idx = np.argmin(np.mean(out[:, :, -2], 1)).item()
                out_prev = out[idx, :, :]
                ft_lr_1 = [0.7,1.,1.4][idx]* ft_learnable_matrix_lrs[idx_ft_lr_1]
                task.ft_lr_1 = ft_lr_1

                if not ft_momentum_params_learnable_lrs[idx_ft_lr_2] == None:
                    out = []
                    for ratio in [0.7,1.,1.4]:
                        ft_lr_2 = ratio * ft_momentum_params_learnable_lrs[idx_ft_lr_2]
                        task.ft_lr_2 = ft_lr_2
                        if ratio == 1.:
                            out.append(out_prev)
                        else:
                            out.append(search())
                    out = np.array(out)  # len, 3, 6
                    idx = np.argmin(np.mean(out[:, :, -2], 1)).item()
                    ft_lr_2 = [0.7,1.,1.4][idx]*ft_momentum_params_learnable_lrs[idx_ft_lr_2]
                    task.ft_lr_2 = ft_lr_2

                out = []
                folder_ = []
                for momentum_params in momentum_params_s:
                    task.momentum_params = momentum_params
                    ser1, ser2 = search(True)
                    out.append(ser1)
                    folder_.append(ser2)
                out = np.array(out)  # len, 3, 6
                idx = np.argmin(np.mean(out[:, :, -2], 1)).item()
                momentum_params = momentum_params_s[idx]
                task.momentum_params = momentum_params
                best_dir = folder_[idx]
                task.write_best_dir(best_dir, out[idx])
                args.start = 1


if args.basic:
    run_basic()
run_ft()