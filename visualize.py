from typing import Optional, List, Union
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt


def vis_1d(data: np.ndarray, title: str = "", save_path: str = "", show: bool = False):
    assert data.ndim == 1
    
    plt.clf()
    plt.plot(data)
    
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


# def vis_input(
#     data: np.ndarray, 
#     var_names: Optional[List[str]] = None,
#     title: str = "",
#     show: bool = False,
#     save_path: str = ""
#     ):
#     assert data.ndim == 2
#     window_size, n_var = data.shape

#     if var_names is not None:
#         assert n_var == len(var_names)
    
#     fig, axes = plt.subplots(nrows=n_var)
#     for idx, ax in enumerate(axes):
#         var_name = var_names[idx] if var_names is not None else None
#         ax.plot(data[:, idx], label=var_name)
#         if var_name is not None:
#             ax.legend(loc='upper right')
    
#     if title:
#         fig.suptitle(title)
    
#     if save_path:
#         plt.savefig(save_path)

#     if show:
#         plt.show()

def vis_input(
    data: Union[np.ndarray, torch.Tensor], 
    indices: Optional[List[int]] = None,
    var_names: Optional[List[str]] = None,
    start: int = 0,
    end: int = -1,
    period: int = 1,
    title: str = "",
    show: bool = False,
    save_path: str = ""
    ):
    assert data.ndim == 2  #! (window_size, n_var)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    if indices is not None:  #! 특정 변수만 선택
        data = data[:, indices]
    
    if end == -1:
        end = len(data)
    data = data[start:end]
    
    if period > 1:
        # data = downsample(data, period)
        start, end = start // period, end // period
    
    window_size, n_var = data.shape

    if var_names is not None:
        assert n_var == len(var_names)
    data_length = len(data[0])  # Assuming data is a 2D array
    fig_width = data_length
    fig, axes = plt.subplots(nrows=n_var, figsize=(fig_width, 9))  
    for idx, ax in enumerate(axes):
        var_name = var_names[idx] if var_names is not None else None
        ax.plot(data[:, idx], label=var_name, linewidth=0.5)
        ax.set_xticks([])  # Hide x-axis ticks
        ax.set_yticks([])  # Hide y-axis ticks
        if var_name is not None:
            ax.legend(loc='upper right')
    
    if title:
        fig.suptitle(title)
    
    if save_path:
        plt.savefig(f"{save_path}.pdf")

    if show:
        plt.show()


def get_data(data_dir, dataset):
    data = pd.read_csv(f"{data_dir}/{dataset}/{dataset}.csv")
    data = data.values[:, 1:]  #! numpy array로 변환하고 date column 제외
    
    return data