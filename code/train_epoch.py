from configuration import CFG
from tqdm import tqdm

def train_one_loop(train_dataloader, model, optimizer, criterion, epoch):
  running_loss = 0.0
  train_batch_cnt = 0

  for data in tqdm(train_dataloader):
    inputs, labels = data
    inputs, labels = inputs.to(CFG.device), labels.to(CFG.device)

    if CFG.use_cutmix == True:
      inputs, labels = cutmix_fn(inputs, labels)

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    train_batch_cnt += 1

  LOGGER.info('Training Epoch: [%d] | loss: %.3f' % (epoch + 1, running_loss/train_batch_cnt))
  return running_loss, train_batch_cnt