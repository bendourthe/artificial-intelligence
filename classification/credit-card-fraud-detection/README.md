___

<a href='http://www.dourthe.tech'> <img src='Dourthe_Technologies_Headers.png' /></a>
___
<center><em>For more information, visit <a href='http://www.dourthe.tech'>www.dourthe.tech</a></em></center>

# Credit Card Fraud Prediction using Logistic Regression

___ 
## Objective
Train a Logistic Regression model to identify fraudulent credit card transactions.

 ___
## Dataset
The datasets contains transactions made by credit cards in September 2013 by european cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

The dataset can be found at: https://www.kaggle.com/mlg-ulb/creditcardfraud

___
## Libraries Imports
### Data manipulation and analysis


```python
import pandas as pd
import numpy as np
```

### Data visualization


```python
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
from jupyterthemes import jtplot
jtplot.style(theme='chesterish')
```

### Machine learning


```python
# Create new synthetic points in an unbalanced dataset using the Synthetic Minority Over-sampling Technique (SMOTE)
from imblearn.over_sampling import SMOTE

# Split data between training and test set
from sklearn.model_selection import StratifiedKFold

# Data scaling
from sklearn.preprocessing import MinMaxScaler

# Create a pipeline for imbalanced datasets
from imblearn.pipeline import make_pipeline

# Support vector machine classifier
from sklearn.linear_model import LogisticRegression

# Cross-validation
from sklearn.model_selection import RandomizedSearchCV

# Model save
import pickle

# Model evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
```

___
## Data Import


```python
df = pd.read_csv('credit_card_fraud.csv')
```

___
## Exploratory Data Analysis
### General

**Display the first few rows of the dataset.**


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



### Missing data
**Check for missing data (expressed as a percentage of the dataset length).**


```python
df.isnull().sum().max()*100/len(df)
```




    0.0



**Thankfully, there are no missing data!**

### Data imbalance

**Let's compare the number of observations for each class.**


```python
sns.countplot('Class', data=df, palette='coolwarm')
plt.title('Number of Regular (0) and Fraudulent (1) transactions')
plt.show()
```


    
![png](img/output_17_0.png)
    


As mentioned in the description of the dataset, it is extremely unbalanced.

**Let's calculate the ratio between classes to better understand the imbalance.**


```python
valuecount = df['Class'].value_counts()*100/len(df)
print('Percentage of non-fraudulent transactions:\t', valuecount[0].round(2), '%')
print('Percentage of fraudulent transactions:\t\t', valuecount[1].round(2), '%')
```

    Percentage of non-fraudulent transactions:    99.83 %
    Percentage of fraudulent transactions:        0.17 %
    

With such a large imbalance, most classifiers would not be able to detect distinguishing patterns between non-fraudulent and fraudulent transactions.

___
## Machine Learning
**To account for the large imbalance of the dataset, we will use cross-validation combined with the Synthetic Minority Over-sampling Technique (SMOTE).**
### Input/Output definition


```python
X = df.drop('Class', axis=1)
y = df['Class']
```

### Train/Test split
**Instead of using the usual train_test_split function from SciKit.Learn, we will use the StratifiedKFold method.**
This cross-validation technique is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.


```python
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

**Compare original label distribution to train/test split label distribution.**


```python
_, original_labels_count = np.unique(y, return_counts=True)
original_dist = np.round(original_labels_count/len(y)*100, 2)
_, train_labels_count = np.unique(y_train, return_counts=True)
train_dist = np.round(train_labels_count/len(y_train)*100, 2)
_, test_labels_count = np.unique(y_test, return_counts=True)
test_dist = np.round(test_labels_count/len(y_test)*100, 2)

print('LABELS DISTRIBUTION')
print('\tOriginal:\t\t', original_dist[0], '% (label 0)\t', original_dist[1], '% (label 1)')
print('\tTraining set:\t\t', train_dist[0], '% (label 0)\t', train_dist[1], '% (label 1)')
print('\tTest set:\t\t', test_dist[0], '% (label 0)\t', test_dist[1], '% (label 1)')
```

    LABELS DISTRIBUTION
      Original:         99.83 % (label 0)   0.17 % (label 1)
      Training set:     99.83 % (label 0)   0.17 % (label 1)
      Test set:         99.83 % (label 0)   0.17 % (label 1)
    

### Data scaling
**To prevent the potential impact of higher weights from variables that have larger ranges of values, we need to scale the data.**

We can start by defining a scaler and fitting it to the training data.

_Warning: do not fit the scaler to the test data, as in a real life scenario we would not be able to fit the scaler to the testing data._


```python
scaler = MinMaxScaler()
```


```python
scaler.fit(X_train)
```




    MinMaxScaler()



**Now, we can transform the training and test data.**

Note: no need to transform the label data.


```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train, y_test = y_train.values, y_test.values
```

### Define and fit model to training data
**First, we need to define a grid of parameters that will be used during cross-validation.**


```python
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
```

**Next, we will define an instance of RandomizedSearchCV using LogisticRegression as the estimator.**

In contrast to GridSearchCV, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions. The number of parameter settings that are tried is given by n_iter. This will allow the training to be a little more efficient.


```python
randomized_logR = RandomizedSearchCV(LogisticRegression(max_iter=300), param_grid, n_iter=4)
```

**Now, we will create a loop that will iterate through each fold generated by the StratifiedKFold method and create a pipeline that will apply the SMOTE method to the dataset and run our defined classifier.**

_Note: a variety of evaluation scores will be saved throughout the process._


```python
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

# Implementing SMOTE Technique 
for train, test in skf.split(X_train_scaled, y_train):
    pipeline = make_pipeline(SMOTE(sampling_strategy='minority'), randomized_logR)
    model = pipeline.fit(X_train_scaled[train], y_train[train])
    best_model = randomized_logR.best_estimator_
    predictions = best_model.predict(X_train_scaled[test])    

    accuracy_lst.append(pipeline.score(X_train_scaled[test], y_train[test]))
    precision_lst.append(precision_score(y_train[test], predictions))
    recall_lst.append(recall_score(y_train[test], predictions))
    f1_lst.append(f1_score(y_train[test], predictions))
    auc_lst.append(roc_auc_score(y_train[test], predictions))
```

**Save the model.**


```python
filename = 'credit_card_fraud_predictor.sav'
pickle.dump(model, open(filename, 'wb'))
```

### Model evaluation
**First, let's print the mean evaluation scores.**


```python
print('MODEL EVALUATION')
print('\tMean accuracy:\t',  np.round(np.mean(accuracy_lst), 2))
print('\tMean precision:\t', np.round(np.mean(precision_lst), 2))
print('\tMean recall:\t', np.round(np.mean(recall_lst), 2))
print('\tMean f1-score:\t', np.round(np.mean(f1_lst), 2))
```

    MODEL EVALUATION
      Mean accuracy:    0.95
      Mean precision:   0.07
      Mean recall:      0.91
      Mean f1-score:    0.12
    

**Then, we will generate predictions using the best model generated during cross-validation:**


```python
predictions = best_model.predict(X_test_scaled)
```

**Evaluate the model using classification report and confusion matrix:**


```python
cm = confusion_matrix(y_test, predictions)
print('CONFUSION MATRIX\n')
print('Total number of')
print('\tTrue positives:\t\t', cm[1,1])
print('\tTrue negatives:\t\t', cm[0,0])
print('\tFalse positives:\t', cm[0,1], '\t\tType I error')
print('\tFalse negatives:\t', cm[1,0], '\t\tType II error')
print('\n')
print('Correct classifications:\t', np.round(100*(cm[0,0]+cm[1,1])/len(X_test),2), '%')
print('Incorrect classifications:\t', np.round(100*(cm[1,0]+cm[0,1])/len(X_test),2), '%')
print('\nCLASSIFICATION REPORT\n')
print(classification_report(y_test, predictions))
```

    CONFUSION MATRIX
    
    Total number of
      True positives:    83
      True negatives:    56184
      False positives:   679    Type I error
      False negatives:   15     Type II error
    
    
    Correct classifications:      98.78 %
    Incorrect classifications:    1.22 %
    
    CLASSIFICATION REPORT
    
                  precision    recall  f1-score   support
    
               0       1.00      0.99      0.99     56863
               1       0.11      0.85      0.19        98
    
        accuracy                           0.99     56961
       macro avg       0.55      0.92      0.59     56961
    weighted avg       1.00      0.99      0.99     56961
    
    

**Finally, we can also look at the average precision-recall score and precision-recall curve:**


```python
average_precision = average_precision_score(y_test, predictions)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
```

    Average precision-recall score: 0.09
    


```python
precision = dict()
recall = dict()
precision, recall, _ = precision_recall_curve(y_test, predictions)

plt.figure(figsize=(8,5))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('PRECISION-RECALL CURVE')
plt.show()
```


    
![png](img/output_48_0.png)
    


___
## Conclusion

The Logistic Regression model performed very well in terms of recall score for both fraudulent and non-fraudulent transactions. Also, most incorrect predictions turned out to be False Positives, which remain much better than False Negatives. Indeed, it is largely preferable to predict a False Positive, which would then allow an agent to perform a deeper investigation and find out whether that transaction was actually fraudulent or not. 

The overall precision of the model could still be improved, either by decreasing the imbalance of the dataset (i.e. by including more fraudulent transactions), or by testing other classification approaches (e.g. Support Vector Machine, Artificial Neural Network).

**This pipeline could easily be transferred to other logistic regression problems.**
