# Image Classification using CNN



## Table of Contents

- [Image Classification using CNN](#image-classification-using-cnn)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
- [Phase 1](#phase-1)
  - [Get to Know the Dataset](#get-to-know-the-dataset)
    - [Statistical Summary for each column](#statistical-summary-for-each-column)
      - [Label](#label)
  - [| Max    | 8.0    |](#-max-----80----)
      - [Feature 1](#feature-1)
  - [| Max    | 685.94 |](#-max-----68594-)
      - [Feature 2](#feature-2)
- [Methods](#methods)
  - [Bagging](#bagging)
    - [Performance on the Dataset](#performance-on-the-dataset)
    - [Random Forest](#random-forest)
    - [Why are thay different?](#why-are-thay-different)
    - [Boosting](#boosting)
    - [AdaBoost with Early Stopping](#adaboost-with-early-stopping)
    - [Steps](#steps)
    - [Stacking](#stacking)
  - [Advantages of Ensemble Learning](#advantages-of-ensemble-learning)
  - [Disadvantages of Ensemble Learning](#disadvantages-of-ensemble-learning)
  - [Applications](#applications)

## Introduction
Ensemble learning is a machine learning paradigm where multiple models, often referred to as "weak learners," are combined to solve a particular computational intelligence problem. The main principle behind ensemble methods is that by combining multiple models, the ensemble can achieve better performance than any single one of the models alone.

# Phase 1

## Get to Know the Dataset



### Statistical Summary for each column

#### Label
---
| Metric | Value  |
|--------|--------|
| Count  | 3031.0 |
| Mean   | 5.65   |
| Std    | 2.12   |
| Min    | 0.0    |
| 25%    | 5.0    |
| 50%    | 6.0    |
| 75%    | 7.0    |
| Max    | 8.0    |
---
#### Feature 1
---
| Metric | Value  |
|--------|--------|
| Count  | 3031.0 |
| Mean   | 359.70 |
| Std    | 198.92 |
| Min    | 7.80   |
| 25%    | 176.64 |
| 50%    | 350.35 |
| 75%    | 543.99 |
| Max    | 685.94 |
---
#### Feature 2
---
| Metric | Value  |
|--------|--------|
| Count  | 3031.0 |
| Mean   | 261.07 |
| Std    | 118.34 |
| Min    | 40.


# Methods
Ensemble methods can be divided into several categories based on how the base models are combined:

## Bagging
**introduction**
Bagging, or Bootstrap Aggregating, involves training multiple instances of the same algorithm on different subsets of the training data and averaging their predictions. The most famous example of bagging is the Random Forest algorithm.

<figure>
        <img src="./Img/Bagging_best.png" alt="optimal estimator">
        <figcaption>Finding the best number of estimators</figcaption>
    </figure>

### Performance on the Dataset
<figure>
        <img src="./Img/Bagging_Visuallize.png" alt="optimal estimator">
        <figcaption>Model visuallization on dataset</figcaption>
    </figure>

1. **Shallow Tree (max_depth=3):**
   The shallow tree learns simpler decision boundaries. It focuses on splitting the data into regions that are separated by only a few features. It might underfit if the dataset is complex but in our case it generalize well because of simpler datasets.

2. **Deep Tree (max_depth=None):**
  The deep tree creates intricate decision boundaries that closely fit the training data, potentially capturing noise and outliers.

3. **Less Aggressive Split (min_samples_split=4):**
   This tree create boundaries where each split requires at least 4 samples, potentially leading to more generalized boundaries compared to smaller min_samples_split values.

4. **Entropy Criterion:**
   The entropy tree aims to find boundaries that reduce entropy (increase purity) at each split.

5. **Moderate Depth (max_depth=6):**
   This tree created somewhat complex boundaries, potentially capturing meaningful interactions in the data without overfitting to noise.

5. **Bagging Classifier with n_estimators=30:**
   Each tree in the ensemble (with bootstrap samples) will focus on different subsets of the data, averaging out individual idiosyncrasies and providing robust predictions.

### Random Forest
Random Forest belongs to the family of decision tree-based algorithms but extends their capabilities to improve predictive performance and reduce overfitting.
<figure>
        <img src="./Img/rf_best.png" alt="optimal estimator">
        <figcaption>Finding the best number of estimators</figcaption>
    </figure>
<figure>
        <img src="./Img/rf_Visuallize.png" alt="optimal estimator">
        <figcaption>Model visuallization on dataset</figcaption>
    </figure>

### Why are thay different?
  **Bootstrap Sampling**:each tree is trained on a random subset of the training data. This means that each tree sees a slightly different perspective of the data, capturing different patterns and relationships.<br>
  **Feature Subsampling**: Random forest uses feature sampling but in our case since there are only 2 features there is not much of difference
### Boosting
Boosting involves sequentially training models, where each model attempts to correct the errors of the previous one. This is done by giving more weight to the instances that were misclassified by earlier models. Examples of boosting algorithms include AdaBoost, Gradient Boosting, and XGBoost.
### AdaBoost with Early Stopping
<figure>
        <img src="./Img/ada_best.png" alt="optimal estimator">
        <figcaption>Finding the best number of estimators</figcaption>
    </figure>
<br>
We implemented adaboost in a way if there are no improvements within a certain number of increases in number of estimators, The algorythm stops and choose the best n_estimator.
<img src="./Img/ada_early.png" alt="optimal estimator">

### Steps
<figure>
        <img src="./Img/ada_steps.png" alt="optimal estimator">
        <figcaption>Steps of Ada model</figcaption>
    </figure>
<br>

1. At first, when the model is weak and the weights are evenly distributed, it focuses on correctly classifying the most challenging examples in the dataset. This approach improves accuracy early on by addressing the most significant errors first.

2. In the next steps, each new learner is trained to correct the mistakes of its predecessors. In the early stages, as more weak learners are added, they contribute different perspectives and improve the model's ability to generalize across the dataset, leading to accuracy improvements.
3. in later iterations, the distribution shifts and new patterns emerge and the model struggles to adapt, leading to accuracy fluctuations.
### Stacking
Stacking, or Stacked Generalization, involves training multiple base models and then using another model, the meta-learner, to combine their predictions. The meta-learner is trained on the outputs of the base learners to make the final prediction.

## Advantages of Ensemble Learning
- **Improved Accuracy**: By combining multiple models, ensembles can achieve higher accuracy and robustness than single models.
- **Reduced Overfitting**: Ensemble methods can reduce overfitting by averaging out biases and variances of the base models.
- **Versatility**: They can be applied to a wide range of machine learning tasks, including classification, regression, and anomaly detection.

## Disadvantages of Ensemble Learning
- **Increased Complexity**: Ensembles can be more complex to design, implement, and interpret compared to single models.
- **Longer Training Times**: Training multiple models can be time-consuming and computationally expensive.
- **Resource Intensive**: Ensembles typically require more memory and computational resources.

## Applications
Ensemble learning is used in various domains such as finance (for credit scoring and stock prediction), medicine (for disease prediction and patient diagnosis), and many other fields where predictive accuracy is crucial.


[def]: #filters-1