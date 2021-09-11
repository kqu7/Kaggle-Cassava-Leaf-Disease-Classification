from utils import seed_everything
from configuration import CFG
from train import train
from inference import infer


if __name__ == '__main__':
  seed_everything()

  if CFG.mode == 'Training':
    train()
  else:
    infer()
