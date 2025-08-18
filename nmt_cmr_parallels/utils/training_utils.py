import torch 

class InverseSquareRootSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(InverseSquareRootSchedule, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        if self.last_epoch < self.warmup_steps:
            scale = last_epoch / self.warmup_steps
        else:
            scale = (self.warmup_steps ** 0.5) / (last_epoch ** 0.5)
        
        return [base_lr * scale for base_lr in self.base_lrs]