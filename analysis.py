import os
import h5py
import pickle as pkl
import yaml
import torch
import pandas as pd
import numpy as np
import scipy as sp
from tqdm import tqdm
import matplotlib.pyplot as plt

import dataset
import lfads

import argparse

# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #
# Constants
# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #

PERF_TABLE_FILENAME = 'performance_table.pkl'
PSD_STAT_FILENAME = 'psd_stats.pkl'
CHECK_KEY = 'ecog'
PAD_PSD = False
N_FFT = 250
NUM_PLOT_TRIALS = 5
# PICK_TRIALS = [661666, 666827]
PICK_TRIAL_START_IDX = [1067350, 1499000]
PICK_TRIAL_FILE_IDX = [36, 36]
NUM_PICK_TRIALS = len(PICK_TRIAL_START_IDX)
TRIAL_FIGSIZE = (14,5)
PICK_TRIAL_FIGSIZE = (2 + (NUM_PICK_TRIALS + 1)*2, 5)
BATCH_STD_THRESH = 0.9
BATCH_AMP_THRESH = 4.0
MED_DIFF_THRESH = 0.1
BANDWIDTH_DB_THRESH = -3.   # (dB), 1/2 power
BANDWIDTH_FREQ_THRESH = 5.  # (Hz)

# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #
# Input Parser
# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #

def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir_path','-m',type=str,required=True)
    parser.add_argument('--dataset_path','-d',type=str,required=True)
    parser.add_argument('--n_block','-b',type=int,default=None)
    parser.add_argument('--task','-t',type=str,default=None)
    parser.add_argument('--overwrite','-o',action='store_true',default=False)
    args = parser.parse_args()
    return args

def parse_model_path(model_dir_path):
    # dir-level information
    model_path_split = model_dir_path.split(os.path.sep)
    model_config = model_path_split[-1]
    model_name = model_path_split[-2]
    dataset_name = model_path_split[-3]
    # basename-level information
    basename_split = model_config.split('_')
    n_block = int(basename_split[-5][6:])
    end_tags = basename_split[-1].split('-')
    task_candidate = end_tags[-2]
    task = task_candidate if task_candidate in ['recon','pred'] else None
    return model_name, dataset_name, n_block, task

# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #
# Dataloader and Model Configuration
# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #

def create_test_dataloader(dataset_path,n_band,task='recon',device='cuda',test_partition=0.1,batch_size=500,check_key=CHECK_KEY):
    
    # get dataset files
    assert task in ['pred','recon']
    trg_dataset_path = dataset_path
    if n_band > 1:
        dataset_basename = os.path.basename(dataset_path)
        src_dataset_basename = dataset_basename[:7] + f'nband{n_band}_' + dataset_basename[7:]
        src_dataset_path = os.path.join(os.path.dirname(dataset_path),src_dataset_basename)
    else:
        src_dataset_path = dataset_path
    assert os.path.exists(src_dataset_path)
    assert os.path.exists(trg_dataset_path)
    src_record = h5py.File(src_dataset_path,'r')
    trg_record = h5py.File(trg_dataset_path,'r')
    _, trg_seq_len, trg_n_ch = trg_record[check_key].shape

    # create dataset object
    if task=='pred':
        ds = dataset.MultiblockEcogTensorPredictionDataset(
            src_record=src_record,
            trg_record=trg_record,
            device=device,
        )
    else:
        ds = dataset.MultiblockEcogTensorDataset_v2(
            data_record=trg_record,
            filt_data_record=src_record,
            n_band=n_band,
            device=device
        )
    
    # create batch dataloader from test partition of dataset
    n_test_trial = len(ds)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=ds,
        sampler=torch.utils.data.BatchSampler(
            np.arange(n_test_trial), # may need to replace with an explicit sampler
            batch_size=batch_size,
            drop_last=False
        )
    )

    return test_dataloader, ds, trg_seq_len, trg_n_ch

def load_lfads_model(model_dir_path, input_size, device):

    assert os.path.exists(model_dir_path)
    with open(os.path.join(model_dir_path,'hyperparameters.yaml')) as yh:
        hparams = yaml.load(yh,Loader=yaml.FullLoader)

    # create model
    if hparams['model_name'] == 'multiblock_lfads_ecog':
        model = lfads.LFADS_Multiblock_Net(
            input_size              = input_size,
            factor_size             = hparams['model']['factor_size'],
            g_encoder_size          = hparams['model']['g_encoder_size'],
            c_encoder_size          = hparams['model']['c_encoder_size'],
            g_latent_size           = hparams['model']['g_latent_size'],
            u_latent_size           = hparams['model']['u_latent_size'],
            controller_size         = hparams['model']['controller_size'],
            generator_size          = hparams['model']['generator_size'],
            n_block                 = hparams['model']['n_block'],
            prior                   = hparams['model']['prior'],
            clip_val                = hparams['model']['clip_val'],
            dropout                 = hparams['model']['dropout'],
            do_normalize_factors    = hparams['model']['normalize_factors'],
            max_norm                = hparams['model']['max_norm'],
            fix_out_mix             = False,
            device                  = device
        )
    elif hparams['model_name'] == 'multiblock_outmix_lfads_ecog':
        model = lfads.LFADS_Multiblock_Net_v2(
            input_size              = input_size,
            factor_size             = hparams['model']['factor_size'],
            g_encoder_size          = hparams['model']['g_encoder_size'],
            c_encoder_size          = hparams['model']['c_encoder_size'],
            g_latent_size           = hparams['model']['g_latent_size'],
            u_latent_size           = hparams['model']['u_latent_size'],
            controller_size         = hparams['model']['controller_size'],
            generator_size          = hparams['model']['generator_size'],
            n_block                 = hparams['model']['n_block'],
            prior                   = hparams['model']['prior'],
            clip_val                = hparams['model']['clip_val'],
            dropout                 = hparams['model']['dropout'],
            do_normalize_factors    = hparams['model']['normalize_factors'],
            max_norm                = hparams['model']['max_norm'],
            device                  = device
        )
    elif hparams['model_name'] == 'multiblock_genmix_lfads_ecog':
        model = lfads.LFADS_Multiblock_GenMix_Net(
            input_size              = input_size,
            factor_size             = hparams['model']['factor_size'],
            g_encoder_size          = hparams['model']['g_encoder_size'],
            c_encoder_size          = hparams['model']['c_encoder_size'],
            g_latent_size           = hparams['model']['g_latent_size'],
            u_latent_size           = hparams['model']['u_latent_size'],
            controller_size         = hparams['model']['controller_size'],
            generator_size          = hparams['model']['generator_size'],
            n_block                 = hparams['model']['n_block'],
            prior                   = hparams['model']['prior'],
            clip_val                = hparams['model']['clip_val'],
            dropout                 = hparams['model']['dropout'],
            do_normalize_factors    = hparams['model']['normalize_factors'],
            max_norm                = hparams['model']['max_norm'],
            device                  = device
        )

    # load from checkpoint
    checkpoint = torch.load(os.path.join(model_dir_path,'checkpoints','best.pth'),map_location=device)
    model.load_state_dict(checkpoint['net'],strict=False)
    model.to(device)
    #TODO: this might break with multidevice models checkpoints. Don't save multidevice models.
    
    return model, hparams['model_name']

# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #
# Test Loop + Model Performance Calculations
# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #

def run_lfads_test_loop(model, model_name, test_dl, fs):
    batch_size = test_dl.sampler.batch_size
    n_trial = len(test_dl.dataset)
    all_trial_idx = np.arange(n_trial)
    model.eval()
    with torch.no_grad():
        performance_metric_table = pd.DataFrame()
        for idx, (src, trg) in tqdm(enumerate(test_dl)):
            
            # get trial idx for this batch, idx of target trial.
            if (idx+1)*batch_size > n_trial:
                trial_idx = all_trial_idx[idx*batch_size:]
            else:
                trial_idx = all_trial_idx[np.arange(batch_size) + idx*batch_size]
            if len(src[0].shape) > 3: # gotta figure out how to fix this
                src = [s.squeeze(dim=0) for s in src]
            if len(trg.shape) > 3:
                trg = trg.squeeze(dim=0)

            #TODO: break this out into different loop functions for each model type
            # ^ side benefit: this will make it easier to xlate to a working ptl implementation.

            # forward pass
            est_dict, (factors, generators, gen_inputs) = model.forward_all(src)

            # pull to cpu
            est, trg, trial_idx = clean_batch_data(
                est_dict['data'].cpu().numpy(),
                trg.cpu().numpy(),
                trial_idx
            )

            # compute signal spectra
            _est_psd, _est_meanch_psd, _trg_psd, _trg_meanch_psd, _err_psd, _err_meanch_psd = compute_trial_psds(est,trg,fs)
            f_psd = np.linspace(0,fs/2,num=_est_psd.shape[1])

            # compute performance/accuracy metrics
            _perf_metrics = compute_performance_metrics(est,trg,trial_idx)
            _psd_metrics = compute_psd_metrics(f_psd,_est_psd,_trg_psd,_err_psd,_est_meanch_psd,_trg_meanch_psd,_err_meanch_psd)
            _perf_metrics = pd.concat([_perf_metrics,_psd_metrics],axis=1)

            # update running table, psd arrays
            performance_metric_table = pd.concat(
                [performance_metric_table, _perf_metrics]
                ,ignore_index=True
            )
            if idx == 0:
                est_psd = _est_psd
                est_meanch_psd = _est_meanch_psd
                trg_psd = _trg_psd
                trg_meanch_psd = _trg_meanch_psd
                err_psd = _err_psd
                err_meanch_psd = _err_meanch_psd
            else:
                est_psd = np.concatenate((est_psd,_est_psd),axis=0)
                est_meanch_psd = np.concatenate((est_meanch_psd,_est_meanch_psd),axis=0)
                trg_psd = np.concatenate((trg_psd,_trg_psd),axis=0)
                trg_meanch_psd = np.concatenate((trg_meanch_psd,_trg_meanch_psd),axis=0)
                err_psd = np.concatenate((err_psd,_err_psd),axis=0)
                err_meanch_psd = np.concatenate((err_meanch_psd,_err_meanch_psd),axis=0)
    psd_dict = {
        'est': {
            'all_ch': est_psd,
            'mean_ch': est_meanch_psd,
        },
        'trg': {
            'all_ch': trg_psd,
            'mean_ch': trg_meanch_psd,
        },
        'err': {
            'all_ch': err_psd,
            'mean_ch': err_meanch_psd,
        },
        'f_psd': f_psd,
        'srate': fs
    }
    return performance_metric_table, psd_dict

def clean_batch_data(est,trg,trial_idx,std_thresh=BATCH_STD_THRESH,amp_thresh=BATCH_AMP_THRESH,med_diff_thresh=MED_DIFF_THRESH):
    std_above_thresh = np.all(trg.std(axis=1) > std_thresh,axis=-1)
    # max_amp_below_thresh = np.all(np.abs(trg) < amp_thresh,axis=(1,2))
    med_abs_diff = np.median(np.abs(np.diff(trg,axis=1)),axis=(1,2)) > med_diff_thresh
    good_channel_trials = std_above_thresh & med_abs_diff
    # one bad channel and YOU'RE OUT! Badge and gun, yada yada.
    est = est[good_channel_trials]
    trg = trg[good_channel_trials]
    trial_idx = trial_idx[good_channel_trials]
    return est, trg, trial_idx

def _compute_performance_metric_dict(est,trg,axis):
    err = trg - est
    # trial stats (est/trg rank)
    est_xch_std = np.nanstd(est - est.mean(axis=-1)[:,:,None],axis=axis)
    trg_xch_std = np.nanstd(trg - trg.mean(axis=-1)[:,:,None],axis=axis)
    err_xch_std = np.nanstd(err - err.mean(axis=-1)[:,:,None],axis=axis)
    # error metrics
    mse = np.nanmean(err**2,axis=axis)
    mae = np.nanmean(np.abs(err),axis=axis)
    corr = np.nanmean(est*trg,axis=axis)/(np.nanstd(est,axis=axis)*np.nanstd(trg,axis=axis))
    rpe = np.sqrt(mse) / np.nanstd(trg, axis=axis) # relative prediction error
    fvu = mse / np.nanvar(trg,axis=axis) # fraction of variance unexplained
    return {
        'est-xch-std': est_xch_std,
        'trg-xch-std': trg_xch_std,
        'err-xch-std': err_xch_std,
        'mse': mse,
        'mae': mae,
        'corr': corr,
        'rpe': rpe,
        'fvu': fvu
    }

def compute_performance_metrics_separate_channels(est,trg):
    axis = 1 # only aggregate values over the time dimension
    perf_metric_dict = _compute_performance_metric_dict(est,trg,axis=axis)
    df_gen = lambda x, key: pd.DataFrame(
        data=x, columns=[f'{key}_ch{ch_idx}' for ch_idx in range(trg.shape[-1])]
    )
    return pd.concat([df_gen(v,k) for (k, v) in perf_metric_dict.items()],axis=1)

def compute_performance_metrics_all_channels(est,trg):
    axis = (1,2) # aggregate values over time, channel dimensions
    return pd.DataFrame(
        _compute_performance_metric_dict(est,trg,axis=axis)
    )

def compute_performance_metrics(est,trg,trial_idx):
    performance_metrics_sep_ch = compute_performance_metrics_separate_channels(est,trg)
    performance_metrics_all_ch = compute_performance_metrics_all_channels(est,trg)
    performance_metrics = pd.concat([performance_metrics_all_ch,performance_metrics_sep_ch],axis=1)
    performance_metrics['trial_idx'] = trial_idx
    return performance_metrics

def _compute_psd_bw(f_psd,est_psd,trg_psd,err_psd,axis=1,db_thresh=BANDWIDTH_DB_THRESH,low_f_thresh=BANDWIDTH_FREQ_THRESH):
    # f_mask = f_psd > low_f_thresh
    # abs_diff_db_psd = 10*np.abs(np.log10(est_psd)-np.log10(trg_psd))
    # abs_diff_db_psd = abs_diff_db_psd.swapaxes(axis,-1)
    norm_err_psd = 10*(np.log10(err_psd) - np.log10(trg_psd))
    f_bin = np.mean(np.diff(f_psd))
    bandwidth = f_bin * np.sum(norm_err_psd < db_thresh, axis=axis)
    # bandstop_idx = np.argmax(np.logical_and(f_mask,abs_diff_db_psd>db_thresh),axis=-1)
    return bandwidth

def compute_psd_metrics_separate_channels(f_psd,est_psd,trg_psd,err_psd):
    est_bw = _compute_psd_bw(f_psd,est_psd,trg_psd,err_psd)
    df_gen = lambda x, key: pd.DataFrame(
        data=x, columns=[f'{key}_ch{ch_idx}' for ch_idx in range(est_psd.shape[-1])]
    )
    return pd.DataFrame(df_gen(est_bw,'bw'))

def compute_psd_metrics_meanch(f_psd,est_psd,trg_psd,err_psd):
    return pd.DataFrame(
        {
            'bw_meanch': _compute_psd_bw(f_psd,est_psd,trg_psd,err_psd)
        }
    )

def compute_psd_metrics(f_psd,est_psd,trg_psd,err_psd,est_meanch_psd,trg_meanch_psd,err_meanch_psd):
    psd_metrics_sep_ch = compute_psd_metrics_separate_channels(f_psd,est_psd,trg_psd,err_psd)
    psd_metrics_meanch = compute_psd_metrics_meanch(f_psd,est_meanch_psd,trg_meanch_psd,err_meanch_psd)
    return pd.concat([psd_metrics_sep_ch,psd_metrics_meanch],axis=1)

def batch_multichannel_psd(x,fs,pad_sample=PAD_PSD,n_fft=N_FFT):
    if pad_sample:
        n_pad = n_fft - x.shape[1]
        n_dim = len(x.shape)
        pad_def = [(0,0)]*n_dim
        pad_def[1] = (0,n_pad)
        if n_pad > 0:
            x_pad = np.pad(x,pad_def,mode='constant',constant_values=0)
        elif n_pad == 0:
            x_pad = x
        else:
            raise(ValueError('sequence_longer than 1000ms (250 samples). Negative pad length computed.'))
    else:
        x_pad = x
    return np.abs(np.fft.rfft(sp.signal.detrend(x_pad,axis=1,type='linear')/fs,axis=1))**2

def batch_multichannel_multitaper_psd(x,fs,df):
    batch_size, seq_len, n_ch = x.shape
    # compute tapers
    seq_time = seq_len/fs
    nw = seq_time*df/2
    n_taper = max(int(2*nw - 1),1) # max window count for this resolution
    tapers = sp.signal.windows.dpss(seq_len,nw,n_taper)
    # compute tapered psds
    es_idx_str = 'ijk,mj->ijkm' # multiply along the time axis
    es_meanch_idx_str = 'ij,mj->ijm'
    tapered_psd = batch_multichannel_psd(np.einsum(es_idx_str,x,tapers),fs).sum(axis=-1)
    meanch_tapered_psd = batch_multichannel_psd(np.einsum(es_meanch_idx_str,x.mean(axis=-1),tapers),fs).sum(axis=-1)
    #TODO: DPSS tapers are orthogonal, but the scale may be incorrect here.
    return tapered_psd, meanch_tapered_psd

def compute_trial_psds(est,trg,fs):
    df = 5
    compute_psd = lambda x: batch_multichannel_multitaper_psd(x,fs,df)
    err = trg - est
    est_psd, est_meanch_psd = compute_psd(est)
    trg_psd, trg_meanch_psd = compute_psd(trg)
    err_psd, err_meanch_psd = compute_psd(err)
    return est_psd, est_meanch_psd, trg_psd, trg_meanch_psd, err_psd, err_meanch_psd

def compute_psd_statistics(psd_dict):
    # compute mean, CI for PSD estimates of all test dataset trials
    gen_psd_stat_dict = lambda x: {
        'mean': x.mean(axis=0),
        'std': x.std(axis=0),
        'min': x.min(axis=0),
        'max': x.max(axis=0),
        'ci95': np.percentile(x,[2.5,97.5],axis=0),
    }
    psd_stat_dict = {}
    for k_psd, v_psd in psd_dict.items():
        if isinstance(v_psd,dict):
            psd_stat_dict[k_psd] = {}
            for k_ch, v_psd in v_psd.items():
                psd_stat_dict[k_psd][k_ch] = gen_psd_stat_dict(v_psd)
        else:
            psd_stat_dict[k_psd] = v_psd

    return psd_stat_dict

def save_performance_metric_data(performance_metric_table_path,performance_metric_table):
    performance_metric_table_path_noext, ext = os.path.splitext(performance_metric_table_path)
    performance_metric_stat_table_path = performance_metric_table_path_noext + '_stat.csv'
    performance_metric_table.to_pickle(performance_metric_table_path)
    performance_metric_stat_table = performance_metric_table.drop('trial_idx',axis=1).describe([.025,.975])
    print(performance_metric_stat_table)
    performance_metric_stat_table.to_csv(performance_metric_stat_table_path)
    print(f'Test dataset performance data saved to: {performance_metric_table_path}')
    print(f'Test dataset performance stats saved to: {performance_metric_stat_table_path}')
    return performance_metric_table_path, performance_metric_stat_table_path

def load_performance_metric_data(performance_metric_table_path):
    print(f'loading performance data from: {performance_metric_table_path}')
    return pd.read_pickle(performance_metric_table_path)

def save_psd_statistics(model_dir_path,psd_stat_dict):
    psd_stat_file_path = os.path.join(model_dir_path,PSD_STAT_FILENAME)
    with open(psd_stat_file_path,'wb') as pf:
        pkl.dump(psd_stat_dict,pf)

def load_psd_statistics(model_dir_path):
    psd_stat_file_path = os.path.join(model_dir_path,PSD_STAT_FILENAME)
    with open(psd_stat_file_path,'rb') as pf:
        return pkl.load(pf)

# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #
# Model Visualization
# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #

def plot_psd_stats(ax,freq,psd_mean,psd_95ci,color,psd_key):
    # get yrange
    ymax = max(10*np.log10(psd_95ci).reshape(-1))
    ymin = 10*np.log10(psd_95ci[0,-1])
    ymax = 10*np.ceil(ymax/10)
    ymin = 10*np.floor(ymin/10)
    ax.fill_between(
        freq,
        10*np.log10(psd_95ci[0,:]),
        10*np.log10(psd_95ci[1,:]),
        color=color,
        alpha=0.3,
        label=f'{psd_key} CI'
    )
    ax.plot(
        freq,
        10*np.log10(psd_mean),
        color=color,
        label=f'{psd_key} mean'
    )
    ax.set_ylim(ymin,ymax)
    ax.set_xlabel('freq. (Hz)')
    ax.set_ylabel('PSD (dB/Hz)')
    ax.set_title('Estimate, Target PSD Distributions')
    ax.legend(loc=0)
    return ax

def plot_model_trial(ax,time,trg,est,trial_idx,label_switch):
    if label_switch:
        ax.plot(time,trg,color='tab:blue',label='trg')
        ax.plot(time,est,color='tab:orange',label='est')
    else:
        ax.plot(time,trg,color='tab:blue')
        ax.plot(time,est,color='tab:orange')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('amplitude (a.u.)')
    ax.set_title(f'Trial {trial_idx}')
    return ax

def plot_model_trials(ax,model,test_dataset,srate,trial_idx,trial_mse):
    trial_sort_idx = np.argsort(trial_idx)
    model.eval()
    with torch.no_grad():
        # doing this the ugly way because inverse indexing is complicated
        src, trg = test_dataset[trial_idx[trial_sort_idx]]
        est_dict, _ = model(src)
        src = [s.cpu().numpy() for s in src]
        trg = trg.cpu().numpy()
        est = est_dict['data'].cpu().numpy()
        trg_meanch = trg.mean(axis=-1)
        est_meanch = est.mean(axis=-1)
        _, seq_len, n_ch = trg.shape
        for idx, (_trg, _est, ax_idx) in enumerate(zip(trg,est,trial_sort_idx)):
            time = np.arange(seq_len)/srate*1000
            ax[0,ax_idx] = plot_model_trial(ax[0,ax_idx],time,_trg+2*np.arange(n_ch),_est+2*np.arange(n_ch),trial_idx[idx],label_switch=False)
            ax[1,ax_idx] = plot_model_trial(ax[1,ax_idx],time,trg_meanch[idx],est_meanch[idx],trial_idx[idx],label_switch=True)
            ax[1,ax_idx].text(0.95, 0.95, f'err = {trial_mse[ax_idx]:0.3f}',horizontalalignment='right',verticalalignment='top',transform=ax[1,ax_idx].transAxes)
            if ax_idx == 0:
                ax[0,ax_idx].legend(loc=0)
                ax[1,ax_idx].legend(loc=0)
    return ax

def get_pick_trial_idx(ds,pick_trial_time,pick_trial_file):
    assert len(pick_trial_time) == len(pick_trial_file)
    pick_trial_idx = []
    for idx in range(len(pick_trial_time)):
        _pick_trial_idx = np.argwhere(
            np.logical_and(ds.data_record['trial_start_idx'][()] == pick_trial_time[idx],
            ds.data_record['dataset_idx'][()] == pick_trial_file[idx])
        )[0,0]
        _pick_trial_idx = np.argwhere(ds.sample_idx[:,0] == _pick_trial_idx)[0,0]
        pick_trial_idx.append(_pick_trial_idx)
    return np.array(pick_trial_idx)

def create_model_performance_figures(model,ds,performance_metric_table,psd_stat_dict,pick_trials=False):
    # plot PSD statistics
    fig_psd, ax_psd = plt.subplots(1,1,dpi=150)
    # est PSD
    psd_key_list = ['trg','est']
    plot_color_list = ['tab:blue','tab:orange']
    for psd_key, color in zip(psd_key_list,plot_color_list):
        ax_psd = plot_psd_stats(
            ax_psd,
            psd_stat_dict['f_psd'],
            psd_stat_dict[psd_key]['mean_ch']['mean'],
            psd_stat_dict[psd_key]['mean_ch']['ci95'],
            color=color,
            psd_key=psd_key
        )
    
    # plot trials (best, median, worst)
    fig_best_trials, ax_best_trials = plt.subplots(2,NUM_PLOT_TRIALS,dpi=150,constrained_layout=True,figsize=TRIAL_FIGSIZE)
    fig_med_trials, ax_med_trials = plt.subplots(2,NUM_PLOT_TRIALS,dpi=150,constrained_layout=True,figsize=TRIAL_FIGSIZE)
    fig_worst_trials, ax_worst_trials = plt.subplots(2,NUM_PLOT_TRIALS,dpi=150,constrained_layout=True,figsize=TRIAL_FIGSIZE)
    if pick_trials:
        fig_pick_trials, ax_pick_trials = plt.subplots(2,NUM_PICK_TRIALS,dpi=150,constrained_layout=True,figsize=PICK_TRIAL_FIGSIZE)
    trial_argsort_idx = np.argsort(performance_metric_table['mse'])
    mse_sorted = np.sort(performance_metric_table['mse'])
    sorted_trial_idx = performance_metric_table['trial_idx'].values[trial_argsort_idx]
    n_all_trials = len(sorted_trial_idx)
    # best: lowest error. worst: highest error
    best_trial_idx = sorted_trial_idx[:NUM_PLOT_TRIALS]
    best_trial_mse = mse_sorted[:NUM_PLOT_TRIALS]
    med_trial_idx = sorted_trial_idx[n_all_trials//2 - NUM_PLOT_TRIALS//2 + np.arange(NUM_PLOT_TRIALS)]
    med_trial_mse = mse_sorted[n_all_trials//2 - NUM_PLOT_TRIALS//2 + np.arange(NUM_PLOT_TRIALS)]
    worst_trial_idx = sorted_trial_idx[-NUM_PLOT_TRIALS:]
    worst_trial_mse = mse_sorted[-NUM_PLOT_TRIALS:]
    pick_trial_idx = get_pick_trial_idx(ds,PICK_TRIAL_START_IDX,PICK_TRIAL_FILE_IDX)
    if pick_trials:
        pick_trial_mse = [performance_metric_table[performance_metric_table['trial_idx'] == pti]['mse'].values[0] for pti in pick_trial_idx]
    ax_best_trials = plot_model_trials(ax_best_trials,model,ds,psd_stat_dict['srate'],best_trial_idx,best_trial_mse)
    ax_med_trials = plot_model_trials(ax_med_trials,model,ds,psd_stat_dict['srate'],med_trial_idx,med_trial_mse)
    ax_worst_trials = plot_model_trials(ax_worst_trials,model,ds,psd_stat_dict['srate'],worst_trial_idx,worst_trial_mse)
    if pick_trials:
        ax_pick_trials = plot_model_trials(ax_pick_trials,model,ds,psd_stat_dict['srate'],pick_trial_idx,pick_trial_mse)

    # collect figures into dict
    fig_dict = {
        'psd_stats': fig_psd,
        'best_trials': fig_best_trials,
        'med_trials': fig_med_trials,
        'worst_trials': fig_worst_trials,
    }
    if pick_trials:
        fig_dict['pick_trials'] = fig_pick_trials

    return fig_dict

def save_model_performance_figures(model_dir_path,fig_dict):
    fig_dir_path = os.path.join(model_dir_path,'figs')
    if not os.path.exists(fig_dir_path):
        os.makedirs(fig_dir_path)
    for k, fig in fig_dict.items():
        fig.savefig(os.path.join(fig_dir_path,f'{k}.png'))
        fig.savefig(os.path.join(fig_dir_path,f'{k}.svg'))
        plt.close(fig=fig)

# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #
# Main
# - - -- --- ----- -------- ------------- -------- ----- --- -- - - #

def main(model_dir_path,dataset_path,n_block,task,overwrite,pick_trials=False):
    _, _, _n_block, _task = parse_model_path(model_dir_path)
    task = _task if _task is not None else task
    n_block = _n_block if _n_block is not None else n_block
    assert task is not None, 'task not parsed from model path or input arguments.'
    assert n_block is not None, 'model block count not parsed from model path or input arguments.'

    # get available device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create test_dataloader, get data parameters
    srate = 250
    test_dataloader, test_dataset, seq_len, n_ch = create_test_dataloader(
        dataset_path = dataset_path,
        n_band = n_block,
        task = task,
        device = 'cuda' if torch.cuda.is_available() else 'cpu')

    # create model, load checkpoint
    model, model_name = load_lfads_model(
        model_dir_path = model_dir_path,
        input_size=n_ch,
        device=device
    )

    # if performance data already computed, in directory:
    performance_metric_table_path = os.path.join(model_dir_path,PERF_TABLE_FILENAME)
    performance_metric_table_file_exists = os.path.exists(performance_metric_table_path)
    if performance_metric_table_file_exists and not overwrite:
        
        # load data
        performance_metric_table = load_performance_metric_data(performance_metric_table_path)
        psd_stat_dict = load_psd_statistics(model_dir_path)
        #TODO: discrepancy between file path and model path inputs when targeting specific files.
        # replace this with constants at the script init for consistent querying
    
    else:
        
        # test loop: compute each batch reconstruction/prediction, compute metrics for each batch, stack to dask array
        performance_metric_table, psd_dict = run_lfads_test_loop(model, model_name, test_dataloader, srate)

        # compute PSD statistics
        psd_stat_dict = compute_psd_statistics(psd_dict)

        # save performance metrics
        save_performance_metric_data(performance_metric_table_path, performance_metric_table)
        save_psd_statistics(model_dir_path, psd_stat_dict)

    # create plots, write to file
    figure_dict = create_model_performance_figures(model, test_dataset, performance_metric_table, psd_stat_dict, pick_trials=pick_trials)
    save_model_performance_figures(model_dir_path,figure_dict)

if __name__ == "__main__":
    args = parse_inputs()
    main(args.model_dir_path,args.dataset_path,args.n_block,args.task,args.overwrite)