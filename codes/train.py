from train_epoch import train_one_loop
from valid_epoch import valid_one_loop
from utils import *
from dataset import *
from model import *
from loss import *
from optimizer import *


def train():
  if CFG.debug == True:
    print('*' * 14)
    print('DEBUGING MODE:')
    print('*' * 14)
  
  train_image_ids = pd.read_csv(CFG.train_data_path\
                                +'train.csv')['image_id'].to_numpy()
  train_labels = pd.read_csv(CFG.train_data_path\
                                +'train.csv')['label'].to_numpy()  
  model = get_model(CFG.model_name)
  criterion = get_criterion()
  optimizer = get_optimizer(model.parameters())
  transform = get_train_transforms()

  if CFG.pretrained:
    print('Using pretrained model: ' + model.model_name)

  # move the training model to the gpu device
  if CFG.device == 'cuda':
    model = model.cuda()
    
  print('Start training...')

  if CFG.use_cutmix == True:
    criterion = CutMixCriterion(criterion)

  for epoch in range(CFG.n_epochs): 
    num_data = train_image_ids.shape[0]
    folds = get_folds()

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
      train_dataloader, valid_dataloader = get_train_valid_dataloaders(train_image_ids, 
                                                                        train_labels, 
                                                                        train_idx, val_idx)
      print(train_dataloader)
      train_loss, train_batch_cnt = train_one_loop(train_dataloader, model, 
                                                   optimizer, criterion, epoch)
      valid_loss, valid_batch_cnt, valid_correct_prediction_cnt = valid_one_loop(
                                                                      valid_dataloader, 
                                                                      model, criterion, 
                                                                      epoch)
      LOGGER.info('Training Fold %d | Epoch %d\n' % (fold_idx, epoch))
      LOGGER.info('Training Loss: %.3f\n' % (train_loss / train_batch_cnt))
      LOGGER.info('Validation Loss: %.3f | Accuracy: %.3f \n\n' % (valid_loss/valid_batch_cnt,
                              valid_correct_prediction_cnt*1.0/len(valid_dataloader.dataset)))

    if CFG.save_model:
        model_state_save_path = CFG.model_save_path+'_fold_'+str(fold_idx)\
                                  +'_epoch_'+str(fold_idx * CFG.n_epochs)+'_'\
                                  +model.model_name+'.pt'
        torch.save(model.state_dict(), model_state_save_path)