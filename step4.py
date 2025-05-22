## Code to read the embeddings and train a model to predict some feature in the data
##
## Note. This is just intended as an illustration of one possible use case of the embeddings,
## and should not be taken to indicate a real and realistic research question. In particular,
## with just N=144 there is not much power to build a prediction model that will be sensitive
## to small differences between cases, and more over, it would be faster to read the
## 144 cases "by hand".

## the code relies on sklearn, which needs to be installed

## Anders Ledberg, 2025 05 19
## anders.ledberg@gmail.com
import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV

## read the anonymized summaries and the embeddings
data=pd.read_parquet("results_of_step3.parquet")
sd=data['summaries']
## assume we have hand-coded a subset of these cases
## for this example I use stance towards the inquiry and code
## this as 1 for positive, and 0 for neutral or negative
## I hand-coded 30 cases, this is way to few to actually train a
## model but it can serve as an example of what was done in the paper

scodes=[0,0,1,1,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,0,1,0,0,0,1,1,1,0,1]


## extract the data corresponding to the manually labeled sample
mdata=data.loc[range(30)]
cols = [f"embeddings_{i}" for i in range(1536)]
X=mdata[cols].to_numpy()
y=np.array(scodes).astype('int')

## in a real application you should find a good value of the penalizing parameter
## (denoted "C" in sklearn) by cross-validation on the data you have coded

## here we will assume a value of 10 is fine (based on the real data)

## next we use this value of C to estimat the model used for predicting
model = LogisticRegression(
    penalty='l2',              # L2 regularization (default)
    C=10,                     # Inverse of regularization strength (smaller = stronger penalty)
    class_weight='balanced',
    solver='lbfgs',            # Recommended solver for multinomial + L2
    max_iter=1000              # Increase if convergence issues
)

model.fit(X, y)

## apply the prediction from the model to the rest of the data
cmdata=data.loc[30:144]
X_test=cmdata[cols].to_numpy()

## generate prediction probabilities for being in the "1" class
pred=model.predict_proba(X_test)[:,1]
plt.hist(pred)
plt.show()

## print some to check model accuracy
## (obviously this should be done more carfully in the real case)

lowp=[i for i,x in enumerate(pred) if x < 0.2 ]
highp=[i for i,x in enumerate(pred) if x > 0.87 ]

print(cmdata['summaries'].iloc[lowp[5]])

print(cmdata['summaries'].iloc[highp[6]])

## we can print the organizations that were deemed positive and not positive

for i in lowp:
    flag=0
    for l in cmdata['summaries'].iloc[i].splitlines():
        if "organization" in l.lower() and flag==0:
            print(l)
            flag=1
        
for i in highp:
    flag=0
    for l in cmdata['summaries'].iloc[i].splitlines():
        if "organization" in l.lower() and flag==0:
            print(l)
            flag=1



