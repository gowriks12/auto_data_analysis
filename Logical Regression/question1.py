import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

Auto = pd.read_csv("Auto.data", na_values=["?"]).dropna()
median = Auto["mpg"].median()
print("Median: {}".format(median))
Auto["mpg01"]=0
# Median column mpg01
for i, rows in Auto.iterrows():
    if rows['mpg'] > median:
        Auto.at[i, 'mpg01'] = 1
print(Auto)

# Box plot to find relation between mpg01 and other features
plot_columns = ['cylinders','displacement','horsepower','weight','acceleration','year','origin']
for col in plot_columns:
    boxplot = Auto.boxplot(column=[col, 'mpg01'])
    plt.show()

# 70% training and 30% testing
training_data, testing_data = train_test_split(Auto, test_size=0.3, random_state=25)
print("Training data")
print(training_data)
print("Testing data")
print(testing_data)

# Performing logistic regression
predictors = ['horsepower', 'cylinders', 'weight', 'displacement']
X = Auto[predictors]
y = Auto['mpg01']
logic_model = sm.Logit(y, X)
result = logic_model.fit()
print(result.summary())

# Predicting using logistic regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, np.ravel(y_train.astype(int)))

y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)
Ys=[]
for i in y_pred_prob:
    Ys.append(i[1])
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# Plotting logistic regression
plot_columns = ['cylinders','displacement','horsepower','weight']
for col in plot_columns:
    sns.lmplot(x=col, y="mpg01", data=Auto.sample(100), logistic=True, ci=None)
    plt.show()

# Confusion matrix
print("Confusion matrix")
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Precision, recall, F-measure and support
print(classification_report(y_test, y_pred))

# ROC
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'g--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()