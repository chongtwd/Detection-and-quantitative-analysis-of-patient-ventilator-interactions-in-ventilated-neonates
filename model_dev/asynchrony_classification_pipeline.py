import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from AbnormalBreathDetectorCNN import *
from ventiliser.GeneralPipeline import GeneralPipeline
import os
from torch.utils.data import Dataset

def run_ventiliser(waveform_fn, data_cols, correction_window = None, flow_unit_converter = lambda x : x, freq = 100,
                   peep = 5.5, flow_thresh = 0.1, t_len = 0.03, f_base = 0, leak_perc_thresh = 0.66,
                   permit_double_cycling = False, insp_hold_length = 0.5, exp_hold_length = 0.05):
    pipeline = GeneralPipeline()
    pipeline.configure(correction_window=correction_window, flow_unit_converter=flow_unit_converter,
                       freq=freq, peep=peep, flow_thresh=flow_thresh, t_len=t_len, f_base=f_base, 
                       leak_perc_thresh=leak_perc_thresh, permit_double_cycling=permit_double_cycling,
                       insp_hold_length=insp_hold_length, exp_hold_length=exp_hold_length)
    pipeline.load_data(waveform_fn, data_cols)
    pipeline.process()

class BreathDataset(Dataset):
    def __init__(self, waveform_fn, data_cols, index_fn, freq=100, breath_max_len=3):
        self.maxlen = int(freq * breath_max_len) # Specify the maximum breath length to consider
        self.idx = pd.read_csv(index_fn)
        self.waveforms = pd.read_csv(waveform_fn, usecols=data_cols).interpolate().values
        self.idx = self.idx[["breath_start", "breath_end"]].values
        self.idx = self.idx[(self.idx[:, 1] - self.idx[:, 0]) <= self.maxlen]
        self.waveforms = torch.tensor(np.stack([np.vstack([(self.waveforms[x[0]:x[1], 1:] - self.waveforms[x[0]:x[1], 1:].mean(axis=0)) / self.waveforms[x[0]:x[1], 1:].std(axis=0), 
                                                           np.zeros(2 * (self.maxlen - x[1] + x[0])).reshape(-1, 2)]).T 
                                                           for x in self.idx], axis=0)).float()
        if torch.cuda.is_available():
            self.waveforms = self.waveforms.cuda()

    def __len__(self):
        return self.idx.shape[0]
    
    def __getitem__(self, i):
        return self.waveforms[i]


if __name__ == "__main__":
    # Edit these variables to point to the directory where the data files are and the models are kept
    data_dir = "../../data"
    model_dir = "model_checkpoints"
    # Download the models from github
    models = {"early cycling" : model_dir + "/CNN_lr0.001_kern5-11-21_dil1_filters128_no_pool_early cyclingepoch=21-Validation_F1_log=0.93.ckpt",
              "early trigger" : model_dir + "/CNN_lr0.001_kern5-11-21_dil1_filters128_no_pool_early triggerepoch=11-Validation_F1_log=0.78.ckpt",
              "expiratory work" : model_dir + "/CNN_lr0.001_kern5-11-21_dil1_filters128_no_pool_expiratory workepoch=61-Validation_F1_log=0.94.ckpt",
              "failed trigger" : model_dir + "/CNN_lr0.001_kern5-11-21_dil1_filters128_no_pool_failed triggerepoch=37-Validation_F1_log=0.91.ckpt",
              "late cycling" : model_dir + "/CNN_lr0.001_kern5-11-21_dil1_filters128_no_pool_late cyclingepoch=36-Validation_F1_log=0.91.ckpt",
              "late trigger" : model_dir + "/CNN_lr0.001_kern5-11-21_dil1_filters128_no_pool_late triggerepoch=13-Validation_F1_log=0.95.ckpt",
              "multiple trigger" : model_dir + "/CNN_lr0.001_kern5-11-21_dil1_filters128_no_pool_multiple triggerepoch=03-Validation_F1_log=0.89.ckpt"}
    
    cuda = torch.cuda.is_available()

    for k, v in models.items():
        if cuda:
            models[k] = AbnormalBreathDetectorCNN.load_from_checkpoint(v)
        else:
            models[k] = AbnormalBreathDetectorCNN.load_from_checkpoint(v, map_location=torch.device("cpu"))
        models[k].eval()
    
    for d in [data_dir + "/" + x for x in os.listdir(data_dir) if os.path.isdir(data_dir + "/" + x)]:
        # Currently identifying waveform files based on this name, but the script can be edited to run on just a list of files
        # If different types of file naming patterns are present
        waveform_fns = [d + "/" + x for x in os.listdir(d) if x.find("fast_Unknown.csv") > -1]
        for w in waveform_fns:
            print(w)
            run_ventiliser(w, [0,1,2], permit_double_cycling=True)  # Can pass in parameters here to set sampling frequency and PEEP etc
            idx_fn = w[:w.find(".csv")] + "_predicted_Breaths_Raw.csv"
            data = BreathDataset(w, [0,1,2], idx_fn)    # Can pass in parameters here to set sampling frequency
            if cuda:
                output = np.hstack([m(data[:]).cpu().detach().numpy() for k, m in models.items()])
            else:
                output = np.hstack([m(data[:]).detach().numpy() for k, m in models.items()])
            output = pd.DataFrame(np.hstack([data.idx, output]), columns=["breath start", "breath end"] + list(models.keys()))
            output["breath start"] = output["breath start"].astype(int)
            output["breath end"] = output["breath end"].astype(int)
            output.to_csv(w[:w.find(".csv")] + "_predicted_Asynchronies.csv", index=False)
            
            
