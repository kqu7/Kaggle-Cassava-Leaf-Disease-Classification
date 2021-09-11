import torch.optim
from configuration import CFG


if CFG.optimizer_name == 'Ranger':
    from pytorch_ranger import Ranger
elif CFG.optimizer_name == 'AdamP':
    from adamp import AdamP
    
if CFG.scheduler_name == 'GradualWarmupSchedulerV2':
    from warmup_scheduler import GradualWarmupScheduler


def get_optimizer(parameters):
    if CFG.optimizer_name == 'Adam':
        if CFG.scheduler_name == 'GradualWarmupSchedulerV2':
            return torch.optim.Adam(parameters, lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
        else:
            return torch.optim.Adam(parameters, lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    elif CFG.optimizer_name == 'AdamW':
        if CFG.scheduler_name == 'GradualWarmupSchedulerV2':
            return torch.optim.AdamW(parameters, lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
        else:
            return torch.optim.Adam(parameters, lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    elif CFG.optimizer_name == 'AdamP':
        if CFG.scheduler_name == 'GradualWarmupSchedulerV2':
            return AdamP(parameters, lr=CFG.lr, weight_decay=CFG.weight_decay)
        else:
            return AdamP(parameters, lr=CFG.lr, weight_decay=CFG.weight_decay)
    elif CFG.optimizer_name == 'Ranger':
        return Ranger(parameters,lr = CFG.lr, alpha=0.5, k=6, N_sma_threshhold=5,
                      betas=(0.95, 0.999), eps=CFG.eps, weight_decay=CFG.weight_decay)
    else:
        pass


def get_scheduler(scheduler_name, optimizer, batches):
    #['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'OneCycleLR', 'GradualWarmupSchedulerV2']
    if scheduler_name == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, epochs=CFG.n_epochs, steps_per_epoch=batches+1, pct_start=0.1)
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.lr_min, last_epoch=-1)
    elif scheduler_name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=0, last_epoch=-1)
    elif scheduler_name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1, patience=1, threshold=0.0001, cooldown=0, min_lr=CFG.lr_min, eps=CFG.eps)
    elif scheduler_name == 'GradualWarmupSchedulerV2':
        return GradualWarmupSchedulerV2(optimizer=optimizer)
    elif scheduler_name == 'LambdaLR':
        return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer)
    else:
        pass