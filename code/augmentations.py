import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from configuration import CFG

def get_train_transforms():
  train_transforms = Compose(
      [
        RandomResizedCrop(CFG.img_size, CFG.img_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.1),
        RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
      ])
  
  return train_transforms

def get_valid_transforms():
  valid_transforms = Compose(
      [
        RandomResizedCrop(CFG.img_size, CFG.img_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
      ])
  
  return valid_transforms

def get_test_transforms():
  test_transforms = Compose(
      [
       ToTensorV2(p=1.0)
      ]
  )
  return test_transforms

def get_heavy_transforms():
  heavy_transforms = Compose(
      [
        RandomResizedCrop(CFG.img_size, CFG.img_size),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.1),
        RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
      ])
  
  return heavy_transforms

def get_light_transforms():
  light_transforms = Compose(
      [
       CenterCrop(CFG.img_size, CFG.img_size),
       ToTensorV2(p=1.0)
      ]
  )
  return light_transforms

# CUTMIX
def cutmix_fn(data, target, alpha=0.3):
    indices         = torch.randperm(data.size(0))
    shuffled_data   = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w   = np.int(W * cut_rat)
    cut_h   = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



class CutMixCriterion(nn.Module):
    def __init__(self, criterion):
        super(CutMixCriterion, self).__init__()
        self.criterion = criterion

    def forward(self, preds, targets):
        targets1 = targets[0]
        targets2 = targets[1]
        lam = targets[2]
        # print('t1', targets1.size())
        # print('t2', targets2.size())
        # print('lam', lam)
        return lam * self.criterion.forward(
            preds, targets1) + (1 - lam) * self.criterion.forward(preds, targets2)