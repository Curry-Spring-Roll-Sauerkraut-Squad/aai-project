# aai-project

## Motivation

## Dataset

### Original Dataset

https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset

The dataset contains 2,700 + images classified into 47 categories.

### Cleaned Dataset

Thanks to Lukas’s work, some obvious outliers that were classified into the wrong category were removed.
Then we get a cleaner dataset with 2300+ images and 47 classes.

### Enhanced Dataset

Thanks to the work of Lukas and Alex, more yoga posture images are added. At the same time, some underrepresented classes are merged, and new poses​ are added. The dataset increases from 2300 to 4000 images, representing 46 classes.

## Approach

- Image Classification
- Usage of Convolutional Neural Networks & Vision Transformer​
- Tried both pretrained and non-pretrained models for accuracy optimization

### 1. CNN

### 2. Train ViT From Scrach

### 3. ViT Based Fine Tuning

Validation accuracy on original dataset(2700+ images, 47 classes)

```
***** eval metrics *****
  epoch                   =        4.0
  eval_accuracy           =     0.9275
  eval_loss               =     0.4354
  eval_runtime            = 0:00:27.82
  eval_samples_per_second =     14.881
  eval_steps_per_second   =      1.869
```

Validation accuracy on cleaned dataset(2300+ images, 47 classes)

```
***** eval metrics *****
  epoch                   =        4.0
  eval_accuracy           =     0.9528
  eval_loss               =     0.3528
  eval_runtime            = 0:00:23.16
  eval_samples_per_second =     14.636
  eval_steps_per_second   =      1.856

```

Validation accuracy on enhanced dataset(4000 images, 46 classes)

```
***** eval metrics *****
  epoch                   =        4.0
  eval_accuracy           =     0.9754
  eval_loss               =     0.1424
  eval_runtime            = 0:01:05.76
  eval_samples_per_second =      8.668
  eval_steps_per_second   =      1.095
```

## Performance Comparison

## Conclusion
