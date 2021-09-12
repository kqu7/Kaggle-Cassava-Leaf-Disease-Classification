# Introduction
Cassava Leaf Disease Classification is a multiclass classification competition held at Kaggle (https://www.kaggle.com/c/cassava-leaf-disease-classification). As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers. However, viral diseases are major sources of poor yields, and to diagonise diseases, farmers require the help of agricultural expers to visually insepct the plants, which is labor-intensive and time-consuming. The objective of this competition is to identify the common diseases with the help of data science and machine learning. 

In this competition, the training dataset includes 21,367 labelled images of Cassava leaves with four categories of diseases or a healthy category. The task is to classify each image into one of the five categories above. 

![intro-img](./image/cassava-leaf-disease.png)

# Summary

## Exploratory Data Analysis
The dataset was unbalanced with 61.9% of training images labelled with CMD (Cassava Mosaic Disease). Dataset augmentation was done by by downsampling the majority class samples and upsampling the minority classes samples through various image scaling and color shifting. The dataset was split into 5 folds using StratifiedKFold. A random weighted sampler was also applied during the training process to increase the number of minority samples being trained. I trained four models (one resnext50 model, and three tf_efficientnet models with different seeds and one with CUTMIX applied while training) with individual 5-fold cross-validation scores as 0.894, 0.898, 0.895, 0.893. 

![data_distribution](./image/data_distribution.png)

## Model
| Model           | CV          |
| ---             | ---         |
| EfficientNet-1   | 0.898     |
| EfficientNet-2   | 0.893     |
| EfficientNet-3 (CUTMIX applied)    | 0.895    |
| ResNext50     | 0.894 |

The ensemble of the above models delivered 90.11% accuray on the public data and 90.01% accuracy on the private data.

![ensemble](./image/cassava_model.jpg)


## Data Augmentation
The training image augmentation was very heavy, including various types of flips, rotations, scale shifting, change of brightness and color saturations, and random coarse dropouts. 

![image-augmentation](./image/augmentation.png)


Heavy image transformation was applied because the training images were collected from different places and during different time in a day. Thus, the training images contained many irrelevant objects that did not help with the classification task. Using heavy image augmentation can make the model more robust to such noise. The validation augmentation included center crop and normalization. At the inference stage, test-time augmentation was applied to increase accuracy.   

# Project Structure
The project has the following structure:
- `code/`: `.py` scripts with training, inference and data processing functions
- `notebook/`: `.ipynb` notebooks for exploratory data analysis, model training and inference
- `data/`: input csv and json data 
- `image/`: the training images (acutal images are not included due to size constraints and can be downloaded from Kaggle: 
    -  [Competition dataset](https://www.kaggle.com/c/cassava-leaf-disease-classification)
    -  [External merged dataset](https://www.kaggle.com/tahsin/cassava-leaf-disease-merged)
- `output/`: model configurations, weights and diagrams exported from notebooks
- `pretrained-model/`: model configurations and weights pretrained on external datasets

# Result
The final ensembled model with 8x TTA scored 90.11% accuracy in the public leaderboard and 90.01% in the private leaderboard, ranked 84 out of 3947 competitors (top 3%) (https://www.kaggle.com/c/cassava-leaf-disease-classification/leaderboard) and was awarded a competition silver medal.
