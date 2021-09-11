from configuration import CFG
from dataset import *
from model import *
from loss import *
from optimizer import *


def infer():
  models = load_pretrained_models()
  test_dataloader = get_test_dataloader()
  test_img_ids, test_pred_labels = [], []

  # Construct for the purpose of testing
  with torch.no_grad():
    for img, img_filename in test_dataloader:
      # Do not use Test Time Inference
      if CFG.use_TTA == False: 
        voting = np.zeros((len(models), CFG.batch_size, CFG.n_classes))
        imgs = np.zeros((CFG.batch_size, 3, CFG.img_size, CFG.img_size))
      # Use Test Time Inference
      else: 
        heavy_transforms = get_heavy_transforms()
        voting = np.zeros((len(models), CFG.n_TTA, CFG.n_classes))
        imgs = np.zeros((CFG.n_TTA, 3, CFG.img_size, CFG.img_size))

        for aug_no in range(CFG.n_TTA):
            img_np = torch.squeeze(img).numpy()
            img_np = img_np.reshape((img_np.shape[1], img_np.shape[2], -1))
            # print('here')
            # print(test_transforms(img=img_np))
            trans_img = heavy_transforms(image=img_np)['image']
            # print(aug_data.size())
            imgs[aug_no, :, :, :] = trans_img.numpy()

        imgs = torch.from_numpy(imgs).to(torch.float32).to(CFG.device)

      # Ensemble models
      for model_idx in range(len(models)):
          model = models[model_idx]
          model = model.to(CFG.device)
          model.eval()            

          logits = model(imgs)
          voting[model_idx, :, :] = F.softmax(logits).cpu().numpy()

      if CFG.use_TTA:
        voting = np.sum(voting, axis=1)/CFG.n_TTA
      voting = np.sum(voting, axis=0)/len(models)

      pred_label = np.argmax(voting)
      # The file name is formatted as img_id.jpeg
      img_id = img_filename[0][:-4] 

      test_img_ids.append(img_id)
      test_pred_labels.append(pred_label)

  # Generate the submission file
  column_header = ['image_id', 'label']
  submission = pd.DataFrame(zip(test_img_ids, test_pred_labels), columns=column_header)
  submission.to_csv(path_or_buf=CFG.output_path, index=False)
