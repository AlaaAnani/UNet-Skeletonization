# UNet-Skeletonization
Skeletonization using UNet architecture from scratch in Python.'


This is a project for the CSCE460301 - Fundamental of Computer Vision (2021 Spring) course at AUC. All copy rights © go to Alaa Anani and Mohamed Ghanem.
Course Link: http://catalog.aucegypt.edu/preview_course_nopop.php?catoid=36&coid=83731


<kbd>![image](https://drive.google.com/uc?export=view&id=1SUVRgAsfHVd9NnRPbD4JoehxfHakWz9A)</kbd>

# Poster 


<kbd>![image](https://drive.google.com/uc?export=view&id=1mb8A7MD_O6uIK9pSzoUg6I6CfvC4K8zl)</kbd>

# Introduction
Object Skeletonization is the process of extracting skeletal, line-like representations of shapes. It provides a very useful tool for geometric shape understanding and minimal shape representation. It also has a wide variety of applications, most notably in anatomical research and activity detection. Several mathematical algorithmic approaches have been developed to solve this problem, and some of them have been proven quite robust. However, a lesser amount of attention has been invested into deep learning solutions for it. In this paper, we use a 2-stage variant of the famous U-Net architecture to split the problem space into two sub-problems: shape minimization and corrective skeleton thinning. Our model produces results that are visually much better than the baseline SkelNetOn model. We propose a new metric, M-CCORR, based on normalized correlation coefficients as an alternative to F1 for this challenge as it solves the problem of class imbalance, managing to recognize skeleton similarity without suffering from F1's over-sensitivity to pixel-shifts.

# Dataset (https://competitions.codalab.org/competitions/21169)
In this work, we focus on the Pixel SkelNetOn dataset containing a total of 1,725 binary images on the PNG format of size 256x256$ pixels. Objects within each image have been pre-segmented and cleaned. They also contain exactly one object per image. The ground truth was generated algorithmically by author of SkelNetOn with some manual intervention to remove unnecessary branches. Note that the results stated in this paper are preliminary (on a reserved subset of the training set) as we do not yet have access to the validation set ground truth since it is an ongoing competition.

# Methodology

## Two-stage Problem Split
In our method, we split the problem space into two consecutive problems handled by two U-Nets that were trained accordingly. The first sub-problem is concerned with condensing or minimizing input shapes which typically results in a thick skeleton-like shape. The second sub-problem expects a bad skeleton and aims to correct as well as thin it. Our training pipeline consists of two such U-Nets in series. Note that the two models are not trained together, that is, the first-stage model is trained on the original data, then the second-stage one is trained on the outputs of stage 1 and the original target skeletons. This is was experimentally observed to produce better results than training both models together.

## Modified U-Net Architecture
Both models have the exact same structure and hyper-parameters. A high-level view of our modified U-Net architecture is shown in the Figure below:


<kbd>![image](https://drive.google.com/uc?export=view&id=1GoEnphG0RKC_9env5H6XApz98jsdlR5-)</kbd>


As briefly explained before, a U-Net consists of a contracting path and an expansive one, separated by a bottleneck region. Each contraction (i.e., down-sampling) block contains two convolutional layers with ReLU activation, ending with a 3x3 stride-2 max pooling layer. On the other hand, each expansive (i.e., up-sampling) block contains one deconvolutional layer followed by two convolutional layers and a concatenation layer. The last concatenation layer combines the last output  



## Weighted Categorical Cross-entropy Loss
Due to the fact that background pixels heavily outnumber skeleton pixels in target images, mistakes in each of them cannot be weighted equally. This is a classic issue of class imbalance which we address in our loss by giving higher weight to misclassifications on the positive class (i.e., skeleton). This incentivizes the model to be more careful with the skeletal region. On that basis, our loss function is a weighted categorical cross-entropy (CCE) computed as:

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}(p_t, p_{r}) = 1 - w*p_t*p_{r}">


where `w` is the class weights <img src="https://render.githubusercontent.com/render/math?math=w_0 \ll w_1, p_t">is the true class (0 for background, 1 for skeleton), and <img src="https://render.githubusercontent.com/render/math?math=p_{r}"> is the predicted probability.

The following is the implementation of the weighted categorical cross-entropy loss into `Keras`:
```python
def weighted_cce(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss
```

# Results
We have experimented with multiple U-Net based architectures using different losses and weights. Although there are differences among the architectures, there are some parameters that are fixed across all experiments. These parameters are shown in the following table:

<kbd>![image](https://drive.google.com/uc?export=view&id=1meiXKV17cY-5VCpoCxbria90tOfSguEp)</kbd>


## New Metric Proposal

Though the _F1_ score is used widely on this dataset as an evaluation metric, one issue that arises with it is that it is very sensitive to small pixel-offset errors. If the skeleton image and the ground truth are partially translated (even by few pixels), the _F1_ score decreases dramatically (check the Figure below for illustration). This pixel-offset sensitivity poorly evaluates good-looking skeletons though they logically match the ground truth. Hence, we propose the usage of template matching as a new translation-aware metric for this problem. We have developed a variant of the normalized cross-correlation coefficient which we call the Matching Cross-correlation (M-CCORR) as defined in the following equation:


<img src="https://render.githubusercontent.com/render/math?math=\textit{M-CCORR}(y_t, y_p)= \frac{max(CCORR(y_t, y_p))}{log_2(\mathcal{D}(y_t, y_p)+2)}">


where <img src="https://render.githubusercontent.com/render/math?math=y_t"> and <img src="https://render.githubusercontent.com/render/math?math=y_p"> are the ground truth and prediction output images respectively and <img src="https://render.githubusercontent.com/render/math?math=\mathcal{D}"> is a function that return the distance between the bounding box center of the ground truth skeleton and that calculated on the predicted skeleton. This denominator ensures that the Matching CCORR-based metric we are proposing is not only aware of pixel-offsets but also of the logical resemblance between the target and prediction.


<kbd>![image](https://drive.google.com/uc?export=view&id=1t67CHAetKVIIIeHyoYu8VzSLufqNBwVY)</kbd>

## Models Scores
| Model       | F1     | M-CCORR |
|-------------|--------|---------|
| One-stage   | 0.4866 | 0.5604  |
| Two-stage   | 0.5968 | 0.6399  |
| Three-stage | 0.5802 | 0.6271  |

## Sample output
<kbd>![image](https://drive.google.com/uc?export=view&id=1SUVRgAsfHVd9NnRPbD4JoehxfHakWz9A)</kbd>

# Code

`two_stage_pipeline.py` and `three_stage_pipeline.py` have the full training and validation pipelines for our experiments. 
Models are instantiated as the following:
```python
model1 = UNet_Thick('unet_thick1', loss=weighted_cce(np.array([1, 25])), load= not TRAIN_1)
model2 = UNet_Thick('unet_thick2', loss=weighted_cce(np.array([1, 25])), load= not TRAIN_2)
```
and their number (stages) can be changed by adding more models ans logically linking their outputs in the rest of the script.



