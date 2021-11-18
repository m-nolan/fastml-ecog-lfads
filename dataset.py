import torch
# from torch._C import dtype, float32
import numpy as np
import scipy.signal as sps
# from torch.utils.data.dataloader import _DataLoaderIter

class LFADS_MultiSession_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_list, device='cpu'):
        super(LFADS_MultiSession_Dataset, self).__init__()
        
        self.data_list   = data_list
        self.device      = device
        self.tensor_list = []
        
        for data in self.data_list:
            self.tensor_list.append(torch.Tensor(data).to(self.device))
            
    def __getitem__(self, ix):
        try:
            return self.tensor_list[ix], ix
        except KeyError:
            raise StopIteration
            
    def __len__(self):
        return len(self.tensor_list)
    
default_collate = torch.utils.data.dataloader._utils.collate.default_collate

class SessionLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, session_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        
        super(SessionLoader, self).__init__(dataset=dataset,
                                            batch_size=session_size,
                                            shuffle=shuffle,
                                            sampler=sampler,
                                            batch_sampler=batch_sampler,
                                            num_workers=num_workers,
                                            collate_fn=collate_fn,
                                            pin_memory=pin_memory,
                                            drop_last=drop_last,
                                            timeout=timeout,
                                            worker_init_fn=worker_init_fn)
        
    def __iter__(self):
        return _SessionLoaderIter(self)
    
# class _SessionLoaderIter(_DataLoaderIter):
    
#     def __init__(self, loader):
#         super(_SessionLoaderIter, self).__init__(loader)
        
#     def __next__(self):
#         x, idx = super(_SessionLoaderIter, self).__next__()
#         x = x.squeeze()
#         setattr(x, 'session', idx)
#         return x,

#-------------------------------------------------------------------
#-------------------------------------------------------------------
class EcogTensorDataset(torch.utils.data.Dataset):
    r"""Pytorch dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, device='cpu', transform=None, transform_mask=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.device = device
        self.transform = transform
        if transform_mask:
            assert len(self.tensors) == len(transform_mask), f'transform_mask length ({len(transform_mask)}) must match number of tensors ({len(tensors)}).'
        else:
            transform_mask = [True] * len(self.tensors) # all-hot mask
        self.transform_mask = transform_mask


    def __getitem__(self, index):
        # get samples
        sample = [tensor[index] for tensor in self.tensors]
        # apply transform
        if self.transform:
            for idx, s in enumerate(sample):
                if self.transform_mask[idx]:
                    sample[idx] = self.transform(s)
        # assign device
        sample = list_or_tuple_recursive_to(sample,self.device)
        return sample

    def __len__(self):
        return self.tensors[0].size(0)

class Hdf5ArrayDataset(torch.utils.data.Dataset):
    r"""Pytorch dataset class reading trial samples from a single [n_trial x ...] hdf5 dataset.

    """

    def __init__(self, h5_dataset, device='cpu', transform=None):
        self.dataset = h5_dataset
        self.device = device
        self.transform = transform
    
    def __getitem__(self, index):
        sample = self.dataset[index,]
        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample).to(self.device)

    def __len__(self):
        return self.dataset.shape[0]

class MultiblockEcogTensorDataset(torch.utils.data.Dataset):
    r'''
    Dataset wrapping: 
        (1) a full-band ECoG sample h5 record
        (2) a multi-band filtered ECoG sample h5 record

    Data samples are returned as a list. The first element is a list of the band-filtered data samples, while the second element is a tensor of the full-band data sample.

    Filtered samples are filtered from the full-band sample in each draw.

    Arguments:
        - data_path (str):      File path to full-band ECoG data record
        - filt_data_path (str): File path to filtered ECoG data record
    '''

    def __init__(self,data_record,filt_data_record,n_band,part_str,device='cpu',dtype=torch.float32):
        self.data_record = data_record
        self.filt_data_record = filt_data_record
        self.n_band = n_band
        self.part_str = part_str
        self.device = device
        self.dtype = dtype
        assert self.part_str in ['train','valid','test'], f'Invalid partition string. {self.part_str} not in [train,valid,test].'
        check_key = self.partition_band_key(self.n_band-1)
        assert check_key in self.filt_data_record.keys(), f'(n_band = {self.n_band}) key {check_key} not found in filt_data_record.'

    def __getitem__(self,index):
        filt_sample_list = []
        for b_idx in range(self.n_band):
            b_key = self.partition_band_key(b_idx)
            filt_sample_list.append(
                torch.tensor(
                    self.filt_data_record[b_key][index,:,:],
                    dtype=self.dtype
                ).to(self.device)
            )
        full_sample = torch.tensor(
            self.data_record[f'{self.part_str}_ecog'][index,:,:],
            dtype=self.dtype
        ).to(self.device)
        return [filt_sample_list,full_sample]
    
    def __len__(self):
        return self.data_record[f'{self.part_str}_ecog'].shape[0]

    def partition_band_key(self,idx):
        return f'band{idx}_{self.part_str}_ecog'

class MultiblockEcogTensorDataset_v2(torch.utils.data.Dataset):
    r'''
    Dataset wrapping: 
        (1) a full-band ECoG sample h5 record
        (2) a multi-band filtered ECoG sample h5 record

    V2: h5 records are no longer created with explicit train/valid/test partitions. That now occurs at the dataset level.

    Data samples are returned as a list. The first element is a list of the band-filtered data samples, while the second element is a tensor of the full-band data sample.

    Filtered samples are filtered from the full-band sample in each draw.

    Arguments:
        - data_path (str):      File path to full-band ECoG data record
        - filt_data_path (str): File path to filtered ECoG data record
    '''

    def __init__(self,data_record,filt_data_record,n_band,device='cpu',dtype=torch.float32):
        if n_band > 1:
            assert data_record['ecog'].shape[-1] == filt_data_record['ecog_band0'].shape[-1]
        self.data_record = data_record
        self.filt_data_record = filt_data_record
        print('aligning dataset samples...')
        self.set_shared_sample_idx()
        self.n_band = n_band
        self.device = device
        self.dtype = dtype

    def __getitem__(self,index):
        full_sample = torch.tensor(
            self.data_record['ecog'][self.sample_idx[index,0],:,:],
            dtype=self.dtype
        ).to(self.device)
        if self.n_band > 1:
            filt_sample_list = [
                torch.tensor(
                    self.filt_data_record[f'ecog_band{b_idx}'][self.sample_idx[index,1],:,:],
                    dtype=self.dtype
                ).to(self.device) for b_idx in range(self.n_band)
            ]
        else:
            filt_sample_list = [full_sample]
        return filt_sample_list, full_sample
    
    def __len__(self):
        return self.sample_idx.shape[0]

    def set_shared_sample_idx(self):
        print('computing intersection...')
        # create tensors of unique IDs for each trial
        data_trial_loc = np.hstack(
            [
                self.data_record['dataset_idx'][()][:,None],
                self.data_record['trial_start_idx'][()][:,None]
            ]
        ).astype(int)
        filt_data_trial_loc = np.hstack(
            [
                self.filt_data_record['dataset_idx'][()][:,None],
                self.filt_data_record['trial_start_idx'][()][:,None]
            ]
        ).astype(int)
        # create 1d view for each row in ^
        _, n_col = data_trial_loc.shape # will always be 2 (?)
        view_def = {
            'names': ['dataset_idx','trial_start_idx'],
            'formats': 2*[data_trial_loc.dtype]
        }
        # compute intersection of both ID sets, also parent array location indices for free (thank god)
        shared_trial_loc, data_loc_idx, filt_data_loc_idx = np.intersect1d(data_trial_loc.view(view_def),filt_data_trial_loc.view(view_def),assume_unique=True,return_indices=True)
        sample_idx = np.hstack([data_loc_idx[:,None],filt_data_loc_idx[:,None]])
        self.sample_idx = sample_idx

def list_or_tuple_recursive_to(x,device):
    if isinstance(x,(list,tuple)):
        x = [list_or_tuple_recursive_to(_x,device) for _x in x]
    else:
        x = x.to(device)
    return x


class MultiblockEcogTensorPredictionDataset(torch.utils.data.Dataset):
    
    """MultiblockEcogTensorPredictionDataset

    Pytorch dataset class returning src and trg tensors or lists of tensors for timeseries prediction tasks, i.e.
    trg <- f(src)

    Uses h5 records as data sources.

    Inputs:
        src_record: [h5 record] hdf5 record containing src time series trials (may be multiband)
        trg_record: [h5 record] hdf5 record containing trg time series trials
        device: [str] device string defining tensor memory location (default: 'cpu')
        dtype: [Type] data type to which loaded tensors are cast (default: torch.float32)
    """

    def __init__(self,src_record,trg_record,device='cpu',dtype=torch.float32):
        self.src_record = src_record
        self.trg_record = trg_record
        if 'ecog' in src_record.keys():
            n_band = 1
            src_trial_shape = src_record['ecog'][0].shape
        else: # multiband dataset
            n_band = len([k for k in src_record.keys() if "ecog_band" in k])
            src_trial_shape = src_record['ecog_band0'][0].shape
        trg_trial_shape = trg_record['ecog'][0].shape
        self.n_band = n_band
        self.src_trial_shape = src_trial_shape
        self.trg_trial_shape = trg_trial_shape
        self.device = device
        self.dtype = dtype
        self.__set_shared_sample_idx__()

    def __set_shared_sample_idx__(self):
        print('computing src/trg trial intersection...')
        # create tensors of unique IDs for each trial
        src_trial_loc = np.hstack(
            [
                self.src_record['dataset_idx'][()][:,None],
                self.src_record['trial_start_idx'][()][:,None]
            ]
        ).astype(int)
        trg_trial_loc = np.hstack(
            [
                self.trg_record['dataset_idx'][()][:,None],
                self.trg_record['trial_start_idx'][()][:,None]
            ]
        ).astype(int)
        # create 1d view for each row in ^
        _, n_col = trg_trial_loc.shape # will always be 2 (?)
        view_def = {
            'names': ['dataset_idx','trial_start_idx'],
            'formats': 2*[trg_trial_loc.dtype]
        }
        # compute intersection of both ID sets, also parent array location indices for free (thank god)
        shared_trial_loc, src_trial_loc_idx, trg_trial_loc_idx = np.intersect1d(src_trial_loc.view(view_def),trg_trial_loc.view(view_def),assume_unique=True,return_indices=True)

        # find all consecutive pairs of trials, create src/trg subsampling indices accordingly
        src_idx, trg_idx = self.__find_src_trg_pairs__(shared_trial_loc['trial_start_idx'],self.src_trial_shape[0])
        sample_idx = np.hstack([src_trial_loc_idx[src_idx,None],trg_trial_loc_idx[trg_idx,None]])
        self.sample_idx = sample_idx

    @staticmethod
    def __find_src_trg_pairs__(trial_loc,trial_len,no_overlap=False):
        trial_loc_pairs = np.hstack([trial_loc[:-1,None],trial_loc[1:,None]])
        trial_idx = np.arange(len(trial_loc))
        trial_idx_pairs = np.hstack([trial_idx[:-1,None],trial_idx[1:,None]])
        trial_pair_is_consecutive = (np.diff(trial_loc_pairs,axis=-1) == trial_len).squeeze()
        if no_overlap:
            print('warning: no_overlap option not implemented in dataset class. returning all consecutive pairs.')
        return trial_idx_pairs[trial_pair_is_consecutive,0], trial_idx_pairs[trial_pair_is_consecutive,1]


    def __getitem__(self,index):
        
        if self.n_band > 1:
            # block sources and targets
            src_sample_list = [
                torch.tensor(
                    self.src_record[f'ecog_band{b_idx}'][self.sample_idx[index,0],:,:],
                    dtype=self.dtype
                ).to(self.device) for b_idx in range(self.n_band)
            ]
            trg_sample_list = [
                torch.tensor(
                    self.src_record[f'ecog_band{b_idx}'][self.sample_idx[index,1],:,:],
                    dtype=self.dtype
                ).to(self.device) for b_idx in range(self.n_band)
            ]
            # total target model
            trg_sample = torch.tensor(
                self.trg_record['ecog'][self.sample_idx[index,1],:,:]
            ).to(self.device)
        else: # n_band = 1
            # block sources and targets
            src_sample_list = [
                torch.tensor(
                    self.src_record['ecog'][self.sample_idx[index,0],:,:],
                    dtype=self.dtype
                ).to(self.device)
            ]
            trg_sample_list = [
                torch.tensor(
                    self.src_record['ecog'][self.sample_idx[index,1],:,:],
                    dtype=self.dtype
                ).to(self.device)
            ]
            # total target model
            trg_sample = trg_sample_list[0]
        return src_sample_list, trg_sample_list, trg_sample

    def __len__(self):
        return self.sample_idx.shape[0]


#-------------------------------------------------------------------
#-------------------------------------------------------------------
# data dropout transforms
class DropChannels(object):
    '''
        Dataset transform to randomly drop channels (i.e. set all values to zero) within a sample.
        The number of dropped channels is determined by the drop ratio:
            n_drop = floor(drop_ratio*n_ch)
        Channel dimension is assumed to be the last indexed tensor dimension. This may need to be
        adjusted for multidimensional time series data, e.g. spectrograms.
    '''
    def __init__(self,drop_ratio=0.1):
        self.drop_ratio = drop_ratio

    def __call__(self,sample):
        n_ch = sample.shape[-1]
        n_ch_drop = np.floor(self.drop_ratio*n_ch)
        drop_ch_idx = torch.randperm(n_ch)[:n_ch_drop]
        sample[:,drop_ch_idx] = 0.
        return sample
    
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# data filtering transform
class FilterData(torch.nn.Module):
    '''
        Dataset transform to filter data samples. Any number of filter bands are allowed. Each filter is interpreted as a bandpass filter requiring 2 corner frequencies.
        A fixed-order IIR filter is used to filter data to each given window.
        Samples are interpreted as a [time x n_ch x ...] tensor and are filtered along the first dimension (pre-batch). 
        Dataset filtered samples are returned in a tuple.

        Inputs:
            - w:    List-like of normalized frequency windows defining each data filter. Each window is defined by a 2-element array or list.
            - n:    IIR filter order.

        Outputs:
            - samples:  List of filtered samples, as defined by w.
    '''

    def __init__(self,w,n,padlen=49,normalize=True):
        super(FilterData, self).__init__()
        self.w = w
        self.n = n
        self.padlen = padlen
        self.normalize = normalize

        # parse w
        # -- add this when you consider using variably-formatted 
        
        # create array of filters
        self.filters = []
        self.btypes = []
        for idx, _w in enumerate(self.w):
            if len(_w) < 2:
                if idx == 0:
                    btype = 'lowpass'
                elif idx == len(self.w) - 1:
                    btype = 'highpass'
            else:
                btype = 'bandpass'
            self.filters.append(
                sps.iirfilter(
                    self.n,
                    _w,
                    btype=btype,
                    rs=60,
                    ftype='butter',     # consider 'bessel' if group delays are an issue
                    output='sos'
                    )
                )
            self.btypes.append(btype)

    def forward(self,sample):
        samples_filt = []
        for idx, f in enumerate(self.filters):
            samples_filt.append(
                torch.tensor(sps.sosfiltfilt( # this type conversion may not be necessary
                    f,
                    sample,
                    axis=0,
                    padlen=self.padlen
                    ).copy(),
                    dtype=torch.float32
                ))
            if self.btypes[idx] == 'lowpass':
                None
            else:
                samples_filt[-1] -= samples_filt[-1].mean(dim=0)
            if self.normalize:
                samples_filt[-1] = tensor_zscore(samples_filt[-1],dim=0)
            
        return samples_filt

# create n uniform filter blocks for the FilterData transform shown above
def create_n_block_w(n_block):
    bandwidth = 1/n_block
    w_c = torch.arange(n_block) * bandwidth
    w_c[0] = 0.01
    w_c = torch.cat([w_c,torch.tensor([0.99])])
    w = []
    for idx in range(n_block):
        w.append([w_c[idx],w_c[idx+1]])
    return w

# z-scoring for tensors in pytorch.
def tensor_zscore(x,dim=0):
    mean = x.mean(dim=dim).expand([50,-1,-1]).permute(1,0,2)
    std = x.std(dim=dim).expand([50,-1,-1]).permute(1,0,2)
    return (x - mean) / std

#-------------------------------------------------------------------
#-------------------------------------------------------------------