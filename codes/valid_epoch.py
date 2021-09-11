from configuration import CFG
from tqdm import tqdm

def valid_one_loop(valid_dataloader, model, criterion, epoch):
    running_loss = 0.0
    correct_prediction_cnt = 0
    val_batch_cnt = 0

    with torch.no_grad():
      for data in tqdm(valid_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(CFG.device), labels.to(CFG.device)

        if CFG.use_cutmix == True:
          inputs, labels = cutmix_fn(inputs, labels)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss
        if CFG.use_cutmix == False:
          correct_prediction_cnt += torch.eq(torch.argmax(outputs, dim=1), labels).sum().item()
        else:
          correct_prediction_cnt = 0.0
        val_batch_cnt += 1 

      LOGGER.info('Validation Epoch [%d] | loss: %.3f | accuracy: %.3f' % (epoch, running_loss / val_batch_cnt,
          correct_prediction_cnt * 1.0 / len(valid_dataloader.dataset)))
      
      return running_loss, val_batch_cnt, correct_prediction_cnt