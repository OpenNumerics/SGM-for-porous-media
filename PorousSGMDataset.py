import torch as pt
from torch.utils.data import Dataset

from typing import Tuple

# Dataset class
class PorousDataset (Dataset):
    def __init__(self, device, dtype):
        super().__init__()
        mask = pt.load('./data/gp_success_multieps.pt', map_location=device, weights_only=True).to(dtype=pt.bool)
        print('Number of Good Data Points:', pt.sum(mask).item())

        parameters = pt.load('./data/parameters_multieps.pt', map_location=device, weights_only=True)[mask,:].to(dtype=dtype)
        c_data = pt.load('./data/c_data_multieps.pt', map_location=device, weights_only=True)[mask,:].to(dtype=dtype)
        phi_data = pt.load('./data/phi_data_multieps.pt', map_location=device, weights_only=True)[mask,:].to(dtype=dtype)
        self.N_samples = int(parameters.shape[0])

        # Normalize the input data
        self.mean_c = pt.mean(c_data)
        self.std_c = pt.std(c_data)
        self.mean_phi = pt.mean(phi_data)
        self.std_phi = pt.std(phi_data)
        self.norm_c_data = (c_data - self.mean_c) / self.std_c
        self.norm_phi_data = (phi_data - self.mean_phi) / self.std_phi

        # Normalize the input parameters
        log_l = pt.log(parameters[:,0])
        self.min_log_l = pt.min(log_l)
        self.max_log_l = pt.max(log_l)
        self.log_l_values = (log_l - self.min_log_l) / (self.max_log_l - self.min_log_l)
        self.min_U0 = pt.min(parameters[:,1])
        self.max_U0 = pt.max(parameters[:,1])
        self.U0_values = (parameters[:,1] - self.min_U0) / (self.max_U0 - self.min_U0)
        self.min_F_right = pt.min(parameters[:,2])
        self.max_F_right = pt.max(parameters[:,2])
        self.F_right_values = (parameters[:,2] - self.min_F_right) / (self.max_F_right - self.min_F_right)

    def __len__(self):
        return self.N_samples
    
    def __getitem__(self, index) -> Tuple[pt.Tensor, pt.Tensor]:
        return pt.cat((self.norm_c_data[index,:], self.norm_phi_data[index,:])), pt.stack((self.log_l_values[index], self.U0_values[index], self.F_right_values[index]))
