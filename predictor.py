#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
#edit in csv path
diabetes = pd.read_csv('../data/diabetes.csv')
diabetes.head()


# In[62]:


diabetes.shape


# In[63]:


diabetes.dtypes


# In[64]:


#data cleaning.
#These catagories should not have 0.
glucoseZero = (diabetes['Glucose'] == 0).any()
bpZero = (diabetes['BloodPressure'] == 0).any()
skinZero = (diabetes['SkinThickness'] == 0).any()
insulinZero = (diabetes['Insulin'] == 0).any()
bmiZero = (diabetes['BMI'] == 0).any()

print(glucoseZero, bpZero, skinZero, insulinZero, bmiZero)
#but they do

#need to replace 0s with medians.

medGlucose = diabetes[diabetes['Glucose'] != 0]['Glucose'].median()
medBP = diabetes[diabetes['BloodPressure'] != 0]['BloodPressure'].median()
medSkin = diabetes[diabetes['SkinThickness'] != 0]['SkinThickness'].median()
medInsulin = diabetes[diabetes['Insulin'] != 0]['Insulin'].median()
medBMI = diabetes[diabetes['BMI'] != 0]['BMI'].median()
#calculated medians excluding the 0s

print(medGlucose, medBP, medSkin, medInsulin, medBMI)


# In[65]:


#replace 0s with median values

diabetes['Glucose'] = diabetes['Glucose'].replace(0,medGlucose)
diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(0,medBP)
diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(0,medSkin)
diabetes['Insulin'] = diabetes['Insulin'].replace(0,medInsulin)
diabetes['BMI'] = diabetes['BMI'].replace(0,medBMI)


# In[74]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[75]:


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=1)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                      #5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,                 #use all available CPU cores
    verbose=1
)


# In[76]:


#Split training and testing set
from sklearn.model_selection import train_test_split

# Split data into train and test sets (80% train, 20% test)
train, test = train_test_split(diabetes, test_size=0.2, random_state=1)


# In[77]:


#predictor columns
predictors = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']


# In[82]:


grid_search.fit(train[predictors], train["Outcome"]) #Evaluate combinations of hyperparameters
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use the best model
best_rf = grid_search.best_estimator_ #best_estimator_ is an instance of RandomForestClassifier with best hyperparameters
preds = best_rf.predict(test[predictors])


# In[ ]:





# In[83]:


from sklearn.metrics import accuracy_score


# In[84]:


accuracy = accuracy_score(test["Outcome"],preds)


# In[85]:


accuracy #.7450980392156863
#how close scores are to target


# In[86]:


combined = pd.DataFrame(dict(actual=test["Outcome"], prediction=preds))


# In[87]:


pd.crosstab(index=combined["actual"],columns=combined["prediction"])


# In[92]:


from sklearn.metrics import precision_score
pscore = precision_score(test["Outcome"],preds) #.7451
print(pscore)
#how close scores are to each other
from sklearn.metrics import confusion_matrix

#Create confusion matrix
cm = confusion_matrix(test["Outcome"], preds)

#Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Diabetes Predictor')
plt.show()


# In[26]:


#Analysis on Relationships

#Pairplot
sns.pairplot(diabetes, hue="Outcome", diag_kind = 'scatter')


# In[88]:


#feature importance
feature_importances = best_rf.feature_importances_

#convert to dataframe for readability
importance_df = pd.DataFrame({'Feature': predictors, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues")
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()


# In[90]:


#ROC Curve

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

X_test = test.drop('Outcome', axis=1)
y_test = test['Outcome']

#Get predicted probabilties
y_probs = best_rf.predict_proba(X_test)[:,1]

#Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

#Plot the ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[28]:


#User input function that takes in user inputs and predicts outcome based on that
def userInput():  
    #user inputs
    uPregnancies = input('Enter Number of Pregnancies:')
    uGlucose = input('Enter Glucose Level:')
    uBloodPressure = input('Enter Blood Pressure Level:')
    uSkinThickness = input('Enter Skin Thickness:')
    uInsulin = input('Enter Insulin Level:')
    uBMI = input('Enter BMI:')
    udpf = input('Enter Diabetes Pedigree Function Level:')
    uage = input('Enter Age:')
    
    #sample data of dataframe using type dictionary
    user_sample = pd.DataFrame({
    'Pregnancies':[uPregnancies],
    'Glucose':[uGlucose],
    'BloodPressure':[uBloodPressure],
    'SkinThickness':[uSkinThickness],
    'Insulin':[uInsulin],
    'BMI':[uBMI],
    'DiabetesPedigreeFunction':[udpf],
    'Age':[uage]})
    
    #prediction
    prediction = best_rf.predict(user_sample)
    
    if (prediction[0]==1):
        predictionresult = "Positive"
    else:
        predictionresult = "Negative"
        
    print("Test:", prediction)    
    print("The prediction based on your data is: ", predictionresult)


userInput()


# In[ ]:


#Room for improvement:
#Treat missing values using strategies like iterative imputer
#Incorporate scaling if appropriate for the model
#Use other models like linear regression, neural network
#Use categorical columns in addition to numerical data
#Inquire into removing outliers

