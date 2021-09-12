import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import timm

from configuration import CFG


def get_model(model_name):
  model = None
  if 'efficientnet' in model_name:
    model = get_efficientnet_model()
  elif 'deit' in model_name:
    model = get_deit_model()
  elif 'vit' in model_name:
    model = get_vit_model()
  elif 'resnext' in model_name:
    model = get_resnext_model()
  elif  'resnet' in model_name:
    model = get_resnet_model()
  else:
    raise ValueError("Invalid model choice")

  
  return CassavaNet(model, model_name)


class CassavaNet(nn.Module):
  def __init__(self, model, model_name):
    super().__init__()
    self.model = model
    cur_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    self.model_name = model_name+'-'+cur_time

  def forward(self, x):
    return self.model(x)
  
  def freeze(self):
    for param in self.model.parameters():
        param.requires_grad = False
        
    if 'efficientnet' in self.model_name:
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    elif self.model_name == 'vit_large_patch16_384' or 'deit_base_patch16_224':
        for param in self.model.head.parameters():
            param.requires_grad = True
    elif 'resnext' in self.model_name:
        for param in self.model.fc.parameters():
            param.requires_grad = True
  
  def unfreeze(self):
    for param in self.model.parameters():
      param.requires_grad = True


def get_resnet_model():
  model = torchvision.models.resnet50(pretrained=CFG.pretrained)
  model.fc = nn.Linear(2048, CFG.n_classes)
  model.to(CFG.device)
  return model

def get_resnext_model():
  model = timm.create_model('resnext50_32x4d', pretrained=CFG.pretrained)
  n_features = model.fc.in_features
  model.fc = nn.Linear(n_features, CFG.n_classes)
  return model

def get_efficientnet_model():
  model = timm.create_model('tf_efficientnet_b4_ns', pretrained=CFG.pretrained)
  n_features = model.classifier.in_features
  model.classifier = nn.Linear(n_features, CFG.n_classes)
  return model

def get_deit_model():
  model = torch.hub.load('facebookresearch/deit:main', 
                                      'deit_base_patch16_384', pretrained=CFG.pretrained)
  n_features = model.head.in_features
  model.head = nn.Linear(n_features, CFG.n_classes)
  return model

def get_vit_model():
  model = timm.create_model('vit_large_patch16_384', pretrained=CFG.pretrained)
  n_features = model.head.in_features
  model.head = nn.Linear(n_features, CFG.n_classes)
  return model

def load_pretrained_models():
  models = []
  count = 0

  for model_fpath in os.listdir(CFG.pretrained_model_path):
    if model_fpath[0] == '.': # Avoid reading file like .DS_Store
      continue
    full_path = CFG.pretrained_model_path+model_fpath

    if not os.path.isdir(full_path) and count in CFG.model_list:
        print("Model Loaded:", model_fpath)
        model_name = model_fpath.split('_f')[0]
        model = get_model(model_name)
        info = torch.load(full_path, map_location=torch.device(CFG.device))
        model.load_state_dict(info)
        models.append(model)
        
    count+=1
  
  return models