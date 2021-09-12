import pandas as pd
from torch.utils.data import Dataset, DataLoader
from augmentations import get_train_transforms, get_valid_transforms, get_test_transforms 

from configuration import CFG
from utils import read_img_from_path


class CassavaTrainDataset(Dataset):
  """ Cassava Leaves Training Dataset """
  def __init__(self, train_img_id, train_img_label, transform=None):
    self.train_img_id = train_img_id
    self.train_img_label = train_img_label
    self.transform = transform
  
  def __len__(self):
    return len(self.train_img_id)

  def __getitem__(self, idx): 
    img_path = CFG.train_img_path + str(self.train_img_id[idx])

    img = read_img_from_path(img_path)
    label = self.train_img_label[idx]

    if self.transform:
      img = self.transform(image=img)['image']
      
    return (img, label)


class CassavaValidDataset(Dataset):
  """ Cassava Leaves Validation Dataset """
  def __init__(self, val_img_id, val_img_label, transform=None):
    self.val_img_id = val_img_id
    self.val_img_label = val_img_label
    self.transform = transform
  
  def __len__(self):
    return len(self.val_img_id)

  def __getitem__(self, idx): 
    img_path = CFG.train_img_path + str(self.val_img_id[idx])

    img = read_img_from_path(img_path)

    if self.transform:
      img = self.transform(image=img)['image']   

    label = self.val_img_label[idx]
    return (img, label)


class CassavaTestDataset(Dataset):
  """ Leaves Test Dataset """
  def __init__(self, img_id, transform=None):
    self.img_id = img_id
    self.transform = transform
  
  def __len__(self):
    return len(self.img_id)

  def __getitem__(self, idx): 
    img_path = CFG.test_img_path + str(self.img_id[idx])
    img = read_img_from_path(img_path)

    if self.transform:
      img = self.transform(image=img)['image']
      
    return img, self.img_id[idx]


def get_train_valid_dataloaders(train_image_ids, train_labels, train_idx, valid_idx):
  train_id, train_label = train_image_ids[train_idx], train_labels[train_idx]
  valid_id, valid_label = train_image_ids[valid_idx], train_labels[valid_idx]

  train_dataset = CassavaTrainDataset(train_id, train_label, get_train_transforms())
  valid_dataset = CassavaValidDataset(valid_id, valid_label, get_valid_transforms()) # TODO: change this?
  
  train_dataloader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, 
                                shuffle=True, num_workers=CFG.num_workers)
  valid_dataloader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, 
                                shuffle=True, num_workers=CFG.num_workers)

  return train_dataloader, valid_dataloader

def get_test_dataloader():
  test_df = pd.read_csv(CFG.test_data_path+"sample_submission.csv")
  test_id = test_df['image_id'].to_numpy()

  test_dataset = CassavaTestDataset(test_id, transform=get_test_transforms())
  test_dataloader = DataLoader(test_dataset, batch_size=CFG.test_batch_size, 
                               shuffle=False, num_workers=CFG.num_workers)
  return test_dataloader
