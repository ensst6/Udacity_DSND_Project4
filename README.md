# Starbucks Customer Promontion Exercise
Machine learning problem-solving project for Udacity Data Scientist Nanodegree

## Background
The project is contained in the Jupyter notebook `Starbucks.ipynb`. The notebook is mostly self-explanatory, so this will be brief.  
Starbucks has data on a customer A/B test where receipt of the promotion was randomized, and the rate of purchase of a particular item was tracked. The dataset contains seven customer-related variables, `V1` - `V7`. We aren't told what characteristics these represent.

The goal is to identify which customers should receive a promotion in the future, with the goal of maximizing two metrics:
- Incremental Response Rate (IRR): difference between proportion of customers in treatment & control groups who make a purchase.    
IRR = (purchases in treatment group)/(customers in treatment group) - (purchases in control group)/(customers in control group)

- Net Incremental Revenue (NIR): The gain in revenue based on how many purchases the treatment group makes compared to the control group, less the cost of sending the promotion. For the exercise, the purchase is a single $10.00 item and the cost of the promotion is $0.15/customer.  
NIR = 10\*(purchases in treatment group) - 0.15\*(customers in treatment group) - 10\*(purchases in control group)

## Development  
The training data are in `training.csv`.  

### Key findings from the cleaning & transformation process:  
1. There were no fields with missing data.  
2. There were no obviously mis-coded variables. `V2` and `V3` are continuous variables, with the others are categorical.
3. There didn't seem to be obvious outliers, although without knowing what the variables actually represent, it's hard to tell.
4. The variable `Promotion` was recoded from `Yes`-`No` to `[1, 0]`.
5. Dummy variables were created for all of the categorical columns (`V1`, `V4-V7`).
6. Creating a target variable: Looking at the forms of IRR and NIR and the problem description, it's obvious we want to maximize purchases. But, to see gains, we have to have more purchasers in the promotion group. Initially I naively assumed we wanted to target the promotion to **anyone** who made a purchase. But, some of these purchases in the non-promotion group may be attributable to other factors, not the promotion. So, I decided to make the target variable `promo+purch`, where 1 represents customers who made a purchase after receiving the promotion, and 0 is everyone else. This is equivalent (to my mind, anyway) to trying to target future promotions to customers like those who've responded to them in the past.   

### Preliminary analyses of the training dataset:
1. The promotion was sent to 50.1% of customers. The p-value comparing proportions in the promotion and control group was 0.507, so the promotion was basically randomly distributed.  
2. A purchase was made by 1.70% of customers who received the promotion, compared to 0.76% of control customers. The p-value comparing these proportions was basically zero, so the promotion does seem to lead to purchases (ignoring other factors).  
3. The IRR for the test dataset was 0.0094, so a tiny increase with the promotion.
4. The NIR was $-2334.60, so actually a net loss due to send the promotion to so many customers who didn't make a purchase.
5. Customers who made a purchase after receiving the promotion are 0.85% of the training dataset.

### Applying machine learning models to the test dataset
For model development, the training data file was split 67%/33% into training and validation sets. This was done because the instructions imply we're not supposed to use the test file until we have a final model to make predictions with. It was split using the `stratify` option in sklearn's `train_test_split` to try to preserve the same proportion of `promo+purch` values in both sets.

#### Univariate graphical exploration and logistic regression
A univariate analysis was done to visualize and explore the contribution of each variable to the likelihood of purchase after receiving a promotion. Distributions of variables were graphed, and their relationship to `promo+purch` analyzed with logistic regression. This will give a preliminary indication of which variables are important, and allow us to possibly shrink the feature set.   
For all the categorical variables, the lowest value was taken as a reference category and left out of the model.

`V1`: values 0, 1, 2, 3. A value of 3 was associated with fewer purchases (p = 0.043) vs the other values  
`V2`: continuous. No association with purchase (p = 0.12)  
`V3`: continuous. Negative values associated with purchase (p = 0.005)  
`V4`: values 1, 2. A value of 2 was associated with purchase (p < 0.0001)  
`V5`: values 1, 2, 3, 4. A value of 2 was associated with purchase (p < 0.0001) vs the other values  
`V6`: values 1, 2, 3, 4. No value was associated with purchase (p = 0.07 to 0.29)  
`V7`: values 1, 2. No association with purchase (p = 0.66)

#### Exploratory multivariate model
To further aid feature selection, the significant variables from the univariate analysis (`V1_3`, `V3`, `V4_2`, `V5_2`) were put into a multivariate model to see if each would still be significant when controlling for the others.  
All 4 variables remained significant (p < 0.0001 to 0.01) so all were retained for machine learning modeling.

#### Machine learning models
The above models were run to test for which parameters to include. Now we'll start assessing them based on metrics.
In addition to NIR & IRR, I decided to look at precision, recall, accuracy, and the confusion matrix. Since there are so few purchases, having too many false positives or negatives will influence NIR & IRR a lot, but maybe not accuracy so much.  

**Note:** your results may vary depending on the train-test split you get when you run the notebook.

Since the model is so imbalanced toward non-purchases (about 120:1), the logistic regression models we've run so far probably won't do a very good job of actually predicting purchasers. To test this, I output the results for the multivariate model run above:
```
confusion matrix:
 [[27910     0]
 [  240     0]]
precision: 0.00000; recall: 0.00000, accuracy: 0.99147
roc_auc: 0.50000
```
As expected, the accuracy is very good, but all the other metrics are terrible. This is because this model just classifies everything as belonging to the majority class, so no purchases are predicted.

There are numerous ways of dealing with imbalanced datasets, including under-sampling, over-sampling, and weighting. You should consider which one is best for your model based on your needs. I experimented with under-sampling and weighting and found basically no difference. Class weighting is built into many of the sklearn estimators, so I'm using that for the results I present here.

A logistic regression model using sklearn's `class_weight = "balanced"` option (where the samples are weighted inversely proportional to the frequency of their class label) produces the following:  
```
confusion matrix:
 [[15322 12588]
 [   64   176]]
precision: 0.01379; recall: 0.73333, accuracy: 0.55055
roc_auc: 0.64116
```
With weighting, we are correctly classifying a fair number of purchases, though we have a lot of false positives.
The results for the Starbucks metrics are:
```
IRR: 0.02040; NIR: 343.45000
```
These are much better than for the baseline dataset, but let's see if we can do better.

One potential improvement would be to add interaction terms to the model. These would account for the level of one variable depending on the level of another, rather than the variables being independent.  

To add interactions, we multiply each variable by every other variable. For our four variables this ends up adding six terms to the model. With the interactions, the logistic regression yields:
```
confusion matrix:
 [[14849 13061]
 [   59   181]]
precision: 0.01367; recall: 0.75417, accuracy: 0.53393
roc_auc: 0.64310
IRR: 0.02022; NIR: 346.25000
```
Slightly lower IRR, slightly higher NIR. Worse precision, which is a key metric since we want to avoid false positives. I'd also be worried about overfitting with so many variables. For those reasons, will continue without interaction terms.

Next, I tried tuning the regularization parameter (`C`) from its default value of 1.0. I used sklearn's `LogisticRegressionCV`, which search values from 10^-4 to 10^4 with cross-validation, and scored the model on precision.
The optimal C-value was 0.0008, but that model didn't improve performance from baseline:
```
confusion matrix:
 [[14887 13023]
 [   62   178]]
precision: 0.01348; recall: 0.74167, accuracy: 0.53517
roc_auc: 0.63753
IRR: 0.01963; NIR: 307.90000
```

There's not much else (at least that I can think of) to do with the logistic regression model to improve it. The baseline model without interactions is still the best.  

Next, let's applying a support vector classifier (SVC). For this, the continuous variable (`V3`) was scaled to mean zero and variance one. Again, we'll used "balanced" class weighting.  
After some manual tuning of the hyperparameters, the best model was:
```
confusion matrix:
 [[14102 13808]
 [   60   180]]
precision: 0.01287; recall: 0.75000, accuracy: 0.50735
roc_auc: 0.62763
IRR: 0.01849; NIR: 247.75000
```
with `C` = 5 and `gamma` = 2. Going very much higher or lower than these manually didn't make a difference, so to tune further let's do a grid search, again using precision as the scoring metric, and the following grid in sklearn's `GridSearchCV`:
```
parameters = [{'C': [2, 3, 5, 7],
               'gamma': [1, 1.5, 2, 2.5],
               'class_weight':  ['balanced']
              }]
```
Results:
```
Best GridSearch Parameters: {'C': 7, 'class_weight': 'balanced', 'gamma': 1}  
confusion matrix:
 [[14292 13618]
 [   63   177]]
precision: 0.01283; recall: 0.73750, accuracy: 0.51400
roc_auc: 0.62479
IRR: 0.01896; NIR: 279.75000
```  
There is improvement here, but maybe the parameter range was too narrow, since the highest `C` and lowest `gamma` were selected. Let's see if higher `C` and lower `gamma` value improve the model further. For these runs, I switched to `HalvingGridSearchCV`, which doesn't do an exhaustive grid search, to save computing time.  
Parameter grid:
```
parameters = [{'C': [8, 16, 32, 64],
               'gamma': [0.1, 0.25, 0.75, 0.9, 0.95],
               'class_weight':  ['balanced']
              }]
```
Results:
```
Best GridSearch Parameters: {'C': 16, 'class_weight': 'balanced', 'gamma': 0.25}
confusion matrix:
 [[13198 14712]
 [   51   189]]
precision: 0.01268; recall: 0.78750, accuracy: 0.47556
roc_auc: 0.63019
IRR: 0.01826; NIR: 242.50000
```
No improvement on the initial tuning run. Logistic regression is still the champ. But let's try to get fancier...

The final model was a gradient boosting algorithm, the vaunted [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html). These algorithms use multiple "weak learners", often shallow decision trees, which may give only fair prediction on their own, but which can give better predictions when combined. At each iteration, mis-classified examples are more heavily weighted so predictions of them improve. This, one would hope, would be ideal for our problem.  
XGBoost doesn't have a "balanced" weighting option, but it does let you provide a weight for the "positive" class (`promo+purch` =1 here). I used the inverse of the proportion of customers who made purchases after promotion (117). It also doesn't allow precision as a scoring metric, so I used area under the ROC curve instead.

XGBoost has numerous parameters that can be tuned. I'm no expert on these, but Prashant Banerjee has a good [guide](https://www.kaggle.com/prashant111/a-guide-on-xgboost-hyperparameters-tuning) on Kaggle. For the first iteration, I used defaults except a couple to make the model more conservative (less likely to overfit). These were `max_delta_step` = 5 and `subsample` = 0.5.  
Results:  
```
confusion matrix:
 [[16739 11171]
 [   99   141]]
precision: 0.01246; recall: 0.58750, accuracy: 0.59964
roc_auc: 0.59362
IRR: 0.01787; NIR: 167.10000
```
Despite all the hype, this is worse than logistic regression and SVC. But that's with no tuning.

Next step was fairly extensive tuning using the article cited above as a guide. Again, I used `HalvingGridSearchCV` to speed things up.  
Parameter grid:
```
parameters = [{'eta': [0.05, 0.1, 0.3],
               'gamma':  [0, 10],
               'max_depth':  [2, 4, 6],
               'min_child_weight':  [0, 10, 100],
               'max_delta_step': [0, 5],
               'subsample': [0.5, 1.0]
              }]
```
Results:
```
Best GridSearch Parameters: {'eta': 0.1, 'gamma': 10, 'max_delta_step': 5, 'max_depth': 2, 'min_child_weight': 100, 'subsample': 1.0}
confusion matrix:
 [[14954 12956]
 [   65   175]]
precision: 0.01333; recall: 0.72917, accuracy: 0.53744
roc_auc: 0.63248
IRR: 0.01925; NIR: 282.55000
```
Tuning improves on the naive XGBoost model, but it's still not as good as logistic regression. And that fit completes in seconds.

## Final Results  
The final test is to read in Starbucks' test data file, predict which customers should receive a promotion, and score the final model based on IRR and NIR.  
The test data are in `Test.csv`. The format of the test file is identical to that of the training file. All that had to be done was to make the four key dummy variables, put them in an X array, and run the test.  
The notebook comes with a pre-coded scoring script `test_results.py`, which also gives back the best value Starbucks got (though not how they got it).  
Drum roll...
```
Nice job!  See how well your strategy worked on our test data below!

Your irr with this strategy is 0.0203.

Your nir with this strategy is 475.60.
We came up with a model with an irr of 0.0188 and an nir of 189.45 on the test set.

 How did you do?
 ```
 Phew. After all that work I was able to beat their benchmarks. Of course, the docs don't say how long the Starbucks candidates get to finish this test, and I probably took waaay longer than they did. It also doesn't say how "tuned" their model was.

 But I did find it interesting (and comforting) that the good old worn out one-eyed teddy bear of my armamentarium, logistic regression, worked so well for this challenge.

## History
Created May 22, 2021

## License  
[Licensed](license.md) under the [MIT License](https://spdx.org/licenses/MIT.html).
