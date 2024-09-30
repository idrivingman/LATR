import torch
from torch import nn


#定义网络时手动罗列层
def naive_init_module(mod):
    if isinstance(mod, nn.Conv2d):
            nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(mod, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(mod.weight, 1)
            nn.init.constant_(mod.bias, 0)  
    elif isinstance(mod ,nn.Linear):
         nn.init.normal_(mod.weight, std=0.01)
    return mod



#定义网络时用squential容器罗列层
def naive_init_module_for_squential(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(mod ,nn.Linear):
         nn.init.normal_(mod.weight, std=0.01)
    return mod

def init_transformer(mod):
    for p in mod.parameters():
        torch.nn.init.normal_(p, mean=0, std=1)
        