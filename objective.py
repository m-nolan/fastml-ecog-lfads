import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import rfft
from torch.nn.parallel.data_parallel import DataParallel
import pdb
from math import log

class Base_Loss(nn.Module):
    def __init__(self, loss_weight_dict, l2_gen_scale=0.0, l2_con_scale=0.0):
        super(Base_Loss, self).__init__()
        self.loss_weights = loss_weight_dict
        self.l2_gen_scale = l2_gen_scale
        self.l2_con_scale = l2_con_scale

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
        
    def forward(self, x_orig, x_recon, model):
        pass
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    def weight_schedule_fn(self, step):
        '''
        weight_schedule_fn(step)
        
        Calculate the KL and L2 regularization weights from the current training step number. Imposes
        linearly increasing schedule on regularization weights to prevent early pathological minimization
        of KL divergence and L2 norm before sufficient data reconstruction improvement. See bullet-point
        4 of section 1.9 in online methods
        
        required arguments:
            - step (int) : training step number
        '''
        
        for key in self.loss_weights.keys():
            # Get step number of scheduler
            weight_step = max(step - self.loss_weights[key]['schedule_start'], 0)
            
            # Calculate schedule weight
            self.loss_weights[key]['weight'] = max(min(self.loss_weights[key]['max'] * weight_step/ self.loss_weights[key]['schedule_dur'], self.loss_weights[key]['max']), self.loss_weights[key]['min'])

    def any_zero_weights(self):
        for key, val in self.loss_weights.items():
            if val['weight'] == 0:
                return True
            else:
                pass
        return False

# - - -- --- ----- -------- ----- --- -- - - #
# - - - Wasserstein Adverserial Error  - - - #
# - - -- --- ----- -------- ----- --- -- - - #
class WAE_Loss(Base_Loss):
    def __init__(self,n_units=512,n_layers=3):
        None
    
    def forward(self,pred,trg):
        loss = 1
        loss_dict = {
            'loss': loss
        }
        return loss, loss_dict

# - - -- --- ----- -------- ----- --- -- - - #
# - - -- log-scale freq-domain error  -- - - #
# - - -- --- ----- -------- ----- --- -- - - #

class Lsfde(Base_Loss):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self,trg,pred):
        trg_hat = torch.fft.rfft(trg,dim=self.axis)
        pred_hat = torch.fft.rfft(pred,dim=self.axis)
        diff_ls = torch.log10(pred_hat) - torch.log10(trg_hat)
        err = torch.abs(diff_ls).sum()
        loss = err
        loss_dict = {
            'log_scale_freq_loss': err,
        }
        return loss, loss_dict

class SVLAE_Loss(Base_Loss):
    def __init__(self, loglikelihood_obs, loglikelihood_deep,
                 loss_weight_dict = {'kl_obs' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0,    'max' : 1.0, 'min' : 0.0},
                                     'kl_deep': {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 2000, 'max' : 1.0, 'min' : 0.0},
                                     'l2'     : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 2000, 'max' : 1.0, 'min' : 0.0},
                                     'recon_deep' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 2000, 'max' : 1.0, 'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(SVLAE_Loss, self).__init__(loss_weight_dict=loss_weight_dict, l2_con_scale=l2_con_scale, l2_gen_scale=l2_gen_scale)
        self.loglikelihood_obs  = loglikelihood_obs
        self.loglikelihood_deep = loglikelihood_deep

    def forward(self, x_orig, x_recon, model):
        kl_obs_weight = self.loss_weights['kl_obs']['weight']
        kl_deep_weight = self.loss_weights['kl_deep']['weight']
        l2_weight = self.loss_weights['l2']['weight']
        recon_deep_weight = self.loss_weights['recon_deep']['weight']
#         pdb.set_trace()

        recon_obs_loss  = -self.loglikelihood_obs(x_orig, x_recon['data'], model.obs_model.generator.calcium_generator.logvar)
        recon_deep_loss = -self.loglikelihood_deep(x_recon['spikes'].permute(1, 0, 2), x_recon['rates'].permute(1, 0, 2))
        recon_deep_loss = recon_deep_weight * recon_deep_loss

        kl_obs_loss = kl_obs_weight * model.obs_model.kl_div()
        kl_deep_loss = kl_deep_weight * model.deep_model.kl_div()

        l2_loss = 0.5 * l2_weight * self.l2_gen_scale * model.deep_model.generator.gru_generator.hidden_weight_l2_norm()

        if hasattr(model.deep_model, 'controller'):
            l2_loss += 0.5 * l2_weight * self.l2_con_scale * model.deep_model.controller.gru_controller.hidden_weight_l2_norm()
            
        loss = recon_obs_loss + recon_deep_loss +  kl_obs_loss + kl_deep_loss + l2_loss
        loss_dict = {'recon_obs'  : float(recon_obs_loss.data),
                     'recon_deep' : float(recon_deep_loss.data),
                     'kl_obs'     : float(kl_obs_loss.data),
                     'kl_deep'    : float(kl_deep_loss.data),
                     'l2'         : float(l2_loss.data),
                     'total'      : float(loss.data)}

        return loss, loss_dict

class LFADS_Wasserstein_Loss(nn.Module):
    '''
    Wasserstein Loss module a la WGAN ()

    Estimates a distribution-specific discriminator for estimating neural signal processes
    '''
    def __init__(self,n_sample,n_ch,n_hidden):
        super(LFADS_Wasserstein_Net,self).__init__()
        self.n_sample = n_sample
        self.n_ch = n_ch
        self.n_hidden = n_hidden

        self.wl_module = nn.Sequential(
            nn.Linear(n_sample*n_ch,n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden,1),
        )

    def forward(self,trg,pred):
        # returns the probability that the sample is from the distribution. Should max at 1?
        d_trg   = self.wl_module(trg.reshape(trg.shape[0],-1))
        d_pred  = self.wl_module(pred.reshape(pred.shape[0],-1))
        return -(d_trg-d_pred)

class LFADS_Loss(Base_Loss):
    def __init__(self, loglikelihood,
                 use_fdl = False,
                 use_tdl = True,
                 use_wl = False,
                 loss_weight_dict= {'kl' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0},
                                    'l2' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(LFADS_Loss, self).__init__(loss_weight_dict=loss_weight_dict, l2_con_scale=l2_con_scale, l2_gen_scale=l2_gen_scale)
        self.loglikelihood = loglikelihood
        self.use_fdl = use_fdl
        self.use_tdl = use_tdl
        
    def freq_domain_loss(self, x_orig, x_recon, min_clamp=-20.0, eps=1e-13):
        # data is [n_batch, n_time, n_ch]
        x_orig_lsp = torch.log10(torch.abs(rfft(x_orig,dim=1))+eps)
        x_recon_lsp = torch.log10(torch.abs(rfft(x_recon,dim=1))+eps)
#         x_orig_lsp = torch.clamp(torch.log10(torch.abs(rfft(x_orig,dim=1))),min=min_clamp)
#         x_recon_lsp = torch.clamp(torch.log10(torch.abs(rfft(x_recon,dim=1))),min=min_clamp)
        if torch.any(torch.isnan(x_orig_lsp)) or torch.any(torch.isnan(x_recon_lsp)):
            breakpoint()
        return F.mse_loss(x_orig_lsp,x_recon_lsp,reduction='sum')/x_orig_lsp.shape[0]
        
    def forward(self, x_orig, x_recon, model):
        kl_weight = self.loss_weights['kl']['weight']
        l2_weight = self.loss_weights['l2']['weight']
        
        recon_loss = -self.loglikelihood(x_orig, x_recon['data'])

        # access model methods/loss terms instead of DataParallel methods
        if isinstance(model,DataParallel):
            model = model.module

        kl_loss = kl_weight * model.kl_div()
        if model.__class__.__name__ == 'LFADS_Ecog_CoRNN_Net':
            l2_loss = 0.5 * l2_weight * self.l2_gen_scale * model.generator.cornn_generator.hidden_weight_l2_norm()
        else:
            l2_loss = 0.5 * l2_weight * self.l2_gen_scale * model.generator.gru_generator.hidden_weight_l2_norm()
    
        if hasattr(model, 'controller'):
            l2_loss += 0.5 * l2_weight * self.l2_con_scale * model.controller.gru_controller.hidden_weight_l2_norm()
            
        loss = kl_loss + l2_loss
        loss_dict = {'kl'    : float(kl_loss.data),
                     'l2'    : float(l2_loss.data)}
        if self.use_tdl:
            try:
                loss += recon_loss
            except:
                breakpoint()
            loss_dict['recon'] = float(recon_loss.data)
        if self.use_fdl:
            recon_fdl = self.freq_domain_loss(x_orig,x_recon['data'])
            loss += recon_fdl
            loss_dict['recon_fdl'] = float(recon_fdl.data)
        loss_dict['total'] = float(loss.data)

        if torch.isinf(loss):
            import matplotlib.pyplot as plt
            breakpoint()

        return loss, loss_dict

# This is sufficiently similar the original LFADS_loss class that it may be appropriate to combine them. It'll look clunky.
class Multiblock_LFADS_Loss(Base_Loss):
    def __init__(self, loglikelihood,
                 use_fdl = False,
                 use_tdl = True,
                 loss_weight_dict= {'kl' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0},
                                    'l2' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(Multiblock_LFADS_Loss, self).__init__(loss_weight_dict=loss_weight_dict, l2_con_scale=l2_con_scale, l2_gen_scale=l2_gen_scale)
        self.loglikelihood = loglikelihood
        self.use_fdl = use_fdl
        self.use_tdl = use_tdl
        
    def freq_domain_loss(self, x_orig, x_recon, min_clamp=-20.0, eps=1e-13):
        # data is [n_batch, n_time, n_ch]
        x_orig_lsp = torch.log10(torch.abs(rfft(x_orig,dim=1))+eps)
        x_recon_lsp = torch.log10(torch.abs(rfft(x_recon,dim=1))+eps)
#         x_orig_lsp = torch.clamp(torch.log10(torch.abs(rfft(x_orig,dim=1))),min=min_clamp)
#         x_recon_lsp = torch.clamp(torch.log10(torch.abs(rfft(x_recon,dim=1))),min=min_clamp)
        if torch.any(torch.isnan(x_orig_lsp)) or torch.any(torch.isnan(x_recon_lsp)):
            breakpoint()
        return F.mse_loss(x_orig_lsp,x_recon_lsp,reduction='sum')/x_orig_lsp.shape[0]
        
    def forward(self, x_orig, x_recon, model):
        kl_weight = self.loss_weights['kl']['weight']
        l2_weight = self.loss_weights['l2']['weight']
        
        # reconstruction loss (final recon, not indiv blocks)
        recon_loss = -self.loglikelihood(x_orig, x_recon['data'])
        if self.use_fdl:
            recon_fdl = self.freq_domain_loss(x_orig,x_recon['data'])
        else:
            recon_fdl = None

        # access model methods/loss terms instead of DataParallel methods
        if isinstance(model,DataParallel):
            model = model.module
        # collect kl, l2 regularizations from blocks
        kl_loss = 0
        l2_loss = 0
        for lb in model.lfads_blocks:
            kl_loss += kl_weight * lb.kl_div()
            l2_loss += 0.5 * l2_weight * self.l2_gen_scale * lb.generator.gru_generator.hidden_weight_l2_norm()
            if hasattr(lb, 'controller'):
                l2_loss += 0.5 * l2_weight * self.l2_con_scale * lb.controller.gru_controller.hidden_weight_l2_norm()
            
        loss = kl_loss + l2_loss
        loss_dict = {'kl'    : float(kl_loss.data),
                     'l2'    : float(l2_loss.data)}
        if self.use_tdl:
            loss += recon_loss
            loss_dict['recon'] = float(recon_loss.data)
        if self.use_fdl:
            loss += recon_fdl
            loss_dict['recon_fdl'] = float(recon_fdl.data)
        loss_dict['total'] = float(loss.data)

        # if torch.isinf(loss):
        #     import matplotlib.pyplot as plt
        #     breakpoint()

        return loss, loss_dict

class Multiblock_LFADS_Loss_Allblocks(Base_Loss):
    def __init__(self, loglikelihood,
                 use_fdl = False,
                 use_tdl = True,
                 loss_weight_dict= {'kl' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0},
                                    'l2' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(Multiblock_LFADS_Loss_Allblocks, self).__init__(loss_weight_dict=loss_weight_dict, l2_con_scale=l2_con_scale, l2_gen_scale=l2_gen_scale)
        self.loglikelihood = loglikelihood
        self.use_fdl = use_fdl
        self.use_tdl = use_tdl
        
    def freq_domain_loss(self, x_orig, x_recon, min_clamp=-20.0, eps=1e-13):
        # data is [n_batch, n_time, n_ch]
        x_orig_lsp = torch.log10(torch.abs(rfft(x_orig,dim=1))+eps)
        x_recon_lsp = torch.log10(torch.abs(rfft(x_recon,dim=1))+eps)
#         x_orig_lsp = torch.clamp(torch.log10(torch.abs(rfft(x_orig,dim=1))),min=min_clamp)
#         x_recon_lsp = torch.clamp(torch.log10(torch.abs(rfft(x_recon,dim=1))),min=min_clamp)
        if torch.any(torch.isnan(x_orig_lsp)) or torch.any(torch.isnan(x_recon_lsp)):
            breakpoint()
        return F.mse_loss(x_orig_lsp,x_recon_lsp,reduction='sum')/x_orig_lsp.shape[0]
        
    def forward(self, x_block, x_orig, x_block_out, x_recon, model):
        kl_weight = self.loss_weights['kl']['weight']
        l2_weight = self.loss_weights['l2']['weight']
        
        # compute error from total reconstruction and each block's recon
        recon_loss = -self.loglikelihood(x_orig, x_recon['data'])
        for b_idx in range(len(x_block)):
            recon_loss += -self.loglikelihood(x_block[b_idx], x_block_out[:,:,:,b_idx])
        if self.use_fdl:
            recon_fdl = self.freq_domain_loss(x_orig,x_recon['data'])
        else:
            recon_fdl = None

        # access model methods/loss terms instead of DataParallel methods
        if isinstance(model,DataParallel):
            model = model.module
        kl_loss = 0
        l2_loss = 0
        for lb in model.lfads_blocks:
            kl_loss += kl_weight * lb.kl_div()
            l2_loss += 0.5 * l2_weight * self.l2_gen_scale * lb.generator.gru_generator.hidden_weight_l2_norm()
            if hasattr(lb, 'controller'):
                l2_loss += 0.5 * l2_weight * self.l2_con_scale * lb.controller.gru_controller.hidden_weight_l2_norm()
            
        loss = kl_loss + l2_loss
        loss_dict = {'kl'    : float(kl_loss.data),
                     'l2'    : float(l2_loss.data)}
        if self.use_tdl:
            loss += recon_loss
            loss_dict['recon'] = float(recon_loss.data)
        if self.use_fdl:
            loss += recon_fdl
            loss_dict['recon_fdl'] = float(recon_fdl.data)
        loss_dict['total'] = float(loss.data)

        if torch.isinf(loss):
            import matplotlib.pyplot as plt
            breakpoint()

        return loss, loss_dict

class Multiblock_LFADS_Loss_OnlyBlocks(Base_Loss):
    def __init__(self, loglikelihood,
                 loss_weight_dict= {'kl' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0},
                                    'l2' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(Multiblock_LFADS_Loss_OnlyBlocks, self).__init__(loss_weight_dict=loss_weight_dict, l2_con_scale=l2_con_scale, l2_gen_scale=l2_gen_scale)
        self.loglikelihood = loglikelihood
        
    def freq_domain_loss(self, x_orig, x_recon, min_clamp=-20.0, eps=1e-13):
        # data is [n_batch, n_time, n_ch]
        x_orig_lsp = torch.log10(torch.abs(rfft(x_orig,dim=1))+eps)
        x_recon_lsp = torch.log10(torch.abs(rfft(x_recon,dim=1))+eps)
#         x_orig_lsp = torch.clamp(torch.log10(torch.abs(rfft(x_orig,dim=1))),min=min_clamp)
#         x_recon_lsp = torch.clamp(torch.log10(torch.abs(rfft(x_recon,dim=1))),min=min_clamp)
        if torch.any(torch.isnan(x_orig_lsp)) or torch.any(torch.isnan(x_recon_lsp)):
            breakpoint()
        return F.mse_loss(x_orig_lsp,x_recon_lsp,reduction='sum')/x_orig_lsp.shape[0]
        
    def forward(self, x_block, x_orig, x_block_out, x_recon, model):
        kl_weight = self.loss_weights['kl']['weight']
        l2_weight = self.loss_weights['l2']['weight']
        
        # compute error from total reconstruction and each block's recon
        total_recon_loss = -self.loglikelihood(x_orig, x_recon['data'])
        block_recon_loss = []
        for b_idx in range(len(x_block)):
            block_recon_loss.append(-self.loglikelihood(x_block[b_idx], x_block_out[:,:,:,b_idx]))

        # compute regularization terms
        # access model methods/loss terms instead of DataParallel methods
        if isinstance(model,DataParallel):
            model = model.module
        kl_loss = 0
        l2_loss = 0
        for lb in model.lfads_blocks:
            kl_loss += kl_weight * lb.kl_div()
            l2_loss += 0.5 * l2_weight * self.l2_gen_scale * lb.generator.gru_generator.hidden_weight_l2_norm()
            if hasattr(lb, 'controller'):
                l2_loss += 0.5 * l2_weight * self.l2_con_scale * lb.controller.gru_controller.hidden_weight_l2_norm()
            
        loss = kl_loss + l2_loss + sum(block_recon_loss)
        loss_dict = {'kl'    : float(kl_loss.data),
                     'l2'    : float(l2_loss.data),
                     'total' : float(loss.data)}
        for bidx, bloss in enumerate(block_recon_loss):
            blosskey = f'block{bidx}'
            loss_dict[blosskey] = float(bloss.data)

        if torch.isinf(loss):
            import matplotlib.pyplot as plt
            breakpoint()

        return loss, loss_dict

class Multiblock_LFADS_Loss_BlockSep(LFADS_Loss):

    def __init__(self, loglikelihood,
                 loss_weight_dict= {'kl' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0},
                                    'l2' : {'weight' : 0.0, 'schedule_dur' : 2000, 'schedule_start' : 0, 'max' : 1.0, 'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(Multiblock_LFADS_Loss_BlockSep, self).__init__(loglikelihood,loss_weight_dict=loss_weight_dict, l2_con_scale=l2_con_scale, l2_gen_scale=l2_gen_scale)
        self.loglikelihood = loglikelihood
        
    def freq_domain_loss(self, x_orig, x_recon, min_clamp=-20.0, eps=1e-13):
        # data is [n_batch, n_time, n_ch]
        x_orig_lsp = torch.log10(torch.abs(rfft(x_orig,dim=1))+eps)
        x_recon_lsp = torch.log10(torch.abs(rfft(x_recon,dim=1))+eps)
#         x_orig_lsp = torch.clamp(torch.log10(torch.abs(rfft(x_orig,dim=1))),min=min_clamp)
#         x_recon_lsp = torch.clamp(torch.log10(torch.abs(rfft(x_recon,dim=1))),min=min_clamp)
        if torch.any(torch.isnan(x_orig_lsp)) or torch.any(torch.isnan(x_recon_lsp)):
            breakpoint()
        return F.mse_loss(x_orig_lsp,x_recon_lsp,reduction='sum')/x_orig_lsp.shape[0]
        
    def forward(self, x_block, x_orig, x_block_out, x_recon, model):
        kl_weight = self.loss_weights['kl']['weight']
        l2_weight = self.loss_weights['l2']['weight']

        if isinstance(model,DataParallel):
            model = model.module

        # compute regularization terms
        loss_dict = {}
        loss_list = []
        for bidx, lb in enumerate(model.lfads_blocks):
            # regularization terms
            kl_loss = kl_weight * lb.kl_div()
            l2_loss = 0.5 * l2_weight * self.l2_gen_scale * lb.generator.gru_generator.hidden_weight_l2_norm()
            if hasattr(lb, 'controller'):
                l2_loss += 0.5 * l2_weight * self.l2_con_scale * lb.controller.gru_controller.hidden_weight_l2_norm()
            # reconstruction loss
            recon_loss = -self.loglikelihood(x_block[bidx], x_block_out[:,:,:,bidx])
            block_loss_dict = {
                'kl':       float(kl_loss.data),
                'l2':       float(l2_loss.data),
                'recon':    float(recon_loss.data),
                'total':    float(kl_loss.data) + float(l2_loss.data) + float(recon_loss.data),
            }
            total_loss = kl_loss + l2_loss + recon_loss
            loss_list.append(total_loss)
            # add to total loss dict
            block_key = f'block{bidx}'
            loss_dict[block_key] = block_loss_dict

        total_recon_loss = -self.loglikelihood(x_orig, x_recon['data'])
        loss_list.append(total_recon_loss)
        # # to expand this to the full reconstruction loss, only append to full reconstruction.
            
        # loss = kl_loss + l2_loss + sum(block_recon_loss)
        # loss_dict = {'kl'    : float(kl_loss.data),
        #              'l2'    : float(l2_loss.data),
        #              'total' : float(loss.data)}
        # for bidx, bloss in enumerate(block_recon_loss):
        #     blosskey = f'block{bidx}'
        #     loss_dict[blosskey] = float(bloss.data)

        return loss_list, loss_dict
    
class Conv_LFADS_Loss(LFADS_Loss):
    
    def __init__(self, loglikelihood,
                 loss_weight_dict= {'kl' : {'weight' : 0.0, 
                                            'schedule_dur' : 2000, 
                                            'schedule_start' : 0, 
                                            'max' : 1.0, 
                                            'min' : 0.0},
                                    'l2' : {'weight' : 0.0, 
                                            'schedule_dur' : 2000, 
                                            'schedule_start' : 0, 
                                            'max' : 1.0, 
                                            'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(Conv_LFADS_Loss, self).__init__(loglikelihood=loglikelihood,
                                              loss_weight_dict=loss_weight_dict,
                                              l2_con_scale=l2_con_scale,
                                              l2_gen_scale=l2_gen_scale)
        
        
    def forward(self, x_orig, x_recon, model):
        kl_weight = self.loss_weights['kl']['weight']
        l2_weight = self.loss_weights['l2']['weight']

        recon_loss = -self.loglikelihood(x_orig, x_recon['data'])
        
        kl_loss = model.lfads.kl_div()
        
        l2_loss = 0.5 * l2_weight * self.l2_gen_scale * model.lfads.generator.gru_generator.hidden_weight_l2_norm()
    
        if hasattr(model.lfads, 'controller'):            
            l2_loss += 0.5 * l2_weight * self.l2_con_scale * model.lfads.controller.gru_controller.hidden_weight_l2_norm()
            
        loss = recon_loss +  kl_loss + l2_loss
        loss_dict = {'recon' : float(recon_loss.data),
                     'kl'    : float(kl_loss.data),
                     'l2'    : float(l2_loss.data),
                     'total' : float(loss.data)}

        return loss, loss_dict
    
class Conv_LFADS_Ecog_Loss(LFADS_Loss):
    
    def __init__(self, loglikelihood,
                 use_fdl = False,
                 loss_weight_dict= {'kl' : {'weight' : 0.0, 
                                            'schedule_dur' : 2000, 
                                            'schedule_start' : 0, 
                                            'max' : 1.0, 
                                            'min' : 0.0},
                                    'l2' : {'weight' : 0.0, 
                                            'schedule_dur' : 2000, 
                                            'schedule_start' : 0, 
                                            'max' : 1.0, 
                                            'min' : 0.0}},
                 l2_con_scale=0.0, l2_gen_scale=0.0):
        
        super(Conv_LFADS_Loss, self).__init__(loglikelihood=loglikelihood,
                                              loss_weight_dict=loss_weight_dict,
                                              l2_con_scale=l2_con_scale,
                                              l2_gen_scale=l2_gen_scale)
        
    def forward(self, x_orig, x_recon, model):
        kl_weight = self.loss_weights['kl']['weight']
        l2_weight = self.loss_weights['l2']['weight']

        recon_loss = -self.loglikelihood(x_orig, x_recon['data'])
        
        recon_fdl = self.freq_domain_loss(x_orig, x_recon['data'])
        
        kl_loss = model.lfads.kl_div()
        
        l2_loss = 0.5 * l2_weight * self.l2_gen_scale * model.lfads.generator.gru_generator.hidden_weight_l2_norm()
    
        if hasattr(model.lfads, 'controller'):            
            l2_loss += 0.5 * l2_weight * self.l2_con_scale * model.lfads.controller.gru_controller.hidden_weight_l2_norm()
            
        loss = recon_loss + recon_fdl +  kl_loss + l2_loss
        loss_dict = {'recon' : float(recon_loss.data),
                     'fdl'   : float(recon_fdl.data),
                     'kl'    : float(kl_loss.data),
                     'l2'    : float(l2_loss.data),
                     'total' : float(loss.data)}
        if self.use_fdl:
            loss += recon_fdl
            loss_dict['recon_fdl'] = float(recon_loss.data)

        return loss, loss_dict
        
class LogLikelihoodPoisson(nn.Module):
    
    def __init__(self, dt=1.0, device='cpu'):
        super(LogLikelihoodPoisson, self).__init__()
        self.dt = dt
        
    def forward(self, k, lam):
#         pdb.set_trace()
        return loglikelihood_poisson(k, lam*self.dt)

class LogLikelihoodPoissonSimple(nn.Module):
    
    def __init__(self, dt=1.0, device='cpu'):
        super(LogLikelihoodPoissonSimple, self).__init__()
        self.dt = dt
    
    def forward(self, k, lam):
        return loglikelihood_poissonsimple(k, lam*self.dt)

class LogLikelihoodPoissonSimplePlusL1(nn.Module):
    
    def __init__(self, dt=1.0, device='cpu'):
        super(LogLikelihoodPoissonSimplePlusL1, self).__init__()
        self.dt = dt
    
    def forward(self, k, lam):
        return loglikelihood_poissonsimple_plusl1(k, lam*self.dt)
    
def loglikelihood_poisson(k, lam):
    '''
    loglikelihood_poisson(k, lam)

    Log-likelihood of Poisson distributed counts k given intensity lam.

    Arguments:
        - k (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
        - lam (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
    '''
    return (k * torch.log(lam) - lam - torch.lgamma(k + 1)).mean(dim=0).sum()

def loglikelihood_poissonsimple_plusl1(k, lam):
    return (k * torch.log(lam) - lam - torch.abs(k)).mean(dim=0).sum()

def loglikelihood_poissonsimple(k, lam):
    return (k * torch.log(lam) - lam).mean(dim=0).sum()

class LogLikelihoodGaussian(nn.Module):
    def __init__(self, mse=True):
        super(LogLikelihoodGaussian, self).__init__()
        self.mse = mse
        
    def forward(self, x, mean, logvar=None):
        if logvar is not None:
            out = loglikelihood_gaussian(x, mean, logvar)
        else:
            if self.mse:
                out = -torch.nn.functional.mse_loss(x, mean, reduction='none')
            else:
                out = -torch.nn.functional.l1_loss(x, mean, reduction='none')
            batch_size = out.shape[0]
            out = torch.nan_to_num(out,posinf=10,neginf=-10).sum()/batch_size # still scaled by n_ch * seq_len
        return out
    
def loglikelihood_gaussian(x, mean, logvar):
    from math import pi
    return -0.5*(log(2*pi) + logvar + ((x - mean).pow(2)/torch.exp(logvar))).mean(dim=0).sum()
        

def kldiv_gaussian_gaussian(post_mu, post_lv, prior_mu, prior_lv):
    '''
    kldiv_gaussian_gaussian(post_mu, post_lv, prior_mu, prior_lv)

    KL-Divergence between a prior and posterior diagonal Gaussian distribution.

    Arguments:
        - post_mu (torch.Tensor): mean for the posterior
        - post_lv (torch.Tensor): logvariance for the posterior
        - prior_mu (torch.Tensor): mean for the prior
        - prior_lv (torch.Tensor): logvariance for the prior
    '''
    klc = 0.5 * (prior_lv - post_lv + torch.exp(post_lv - prior_lv) \
         + ((post_mu - prior_mu)/torch.exp(0.5 * prior_lv)).pow(2) - 1.0).mean(dim=0).sum()
    return klc
