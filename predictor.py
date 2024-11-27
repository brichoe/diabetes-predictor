import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#%matplotlib inline

def load_data(file_path):
    return pd.read_csv(file_path)

# User input function that takes in user inputs and predicts outcome based on that
def userInput():
    # user inputs
    uPregnancies = input('Enter Number of Pregnancies:')
    uGlucose = input('Enter Glucose Level:')
    uBloodPressure = input('Enter Blood Pressure Level:')
    uSkinThickness = input('Enter Skin Thickness:')
    uInsulin = input('Enter Insulin Level:')
    uBMI = input('Enter BMI:')
    udpf = input('Enter Diabetes Pedigree Function Level:')
    uage = input('Enter Age:')

    # sample data of dataframe using type dictionary
    user_sample = pd.DataFrame({
        'Pregnancies': [uPregnancies],
        'Glucose': [uGlucose],
        'BloodPressure': [uBloodPressure],
        'SkinThickness': [uSkinThickness],
        'Insulin': [uInsulin],
        'BMI': [uBMI],
        'DiabetesPedigreeFunction': [udpf],
        'Age': [uage]})

    # prediction
    prediction = rf.predict(user_sample)

    if (prediction[0] == 1):
        predictionresult = "Positive"
    else:
        predictionresult = "Negative"

    print("Test:", prediction)
    print("The prediction based on your data is: ", predictionresult)






#main function
if __name__ == "__main__":
    diabetes = load_data("data/diabetes.csv")
    # data cleaning.
    # These catagories should not have 0.
    glucoseZero = (diabetes['Glucose'] == 0).any()
    bpZero = (diabetes['BloodPressure'] == 0).any()
    skinZero = (diabetes['SkinThickness'] == 0).any()
    insulinZero = (diabetes['Insulin'] == 0).any()
    bmiZero = (diabetes['BMI'] == 0).any()

    print(glucoseZero, bpZero, skinZero, insulinZero, bmiZero)
    # but they do

    # need to replace 0s with medians.

    medGlucose = diabetes[diabetes['Glucose'] != 0]['Glucose'].median()
    medBP = diabetes[diabetes['BloodPressure'] != 0]['BloodPressure'].median()
    medSkin = diabetes[diabetes['SkinThickness'] != 0]['SkinThickness'].median()
    medInsulin = diabetes[diabetes['Insulin'] != 0]['Insulin'].median()
    medBMI = diabetes[diabetes['BMI'] != 0]['BMI'].median()
    # calculated medians excluding the 0s

    #print(medGlucose, medBP, medSkin, medInsulin, medBMI)

    # replace 0s with median values

    diabetes['Glucose'] = diabetes['Glucose'].replace(0, medGlucose)
    diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(0, medBP)
    diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(0, medSkin)
    diabetes['Insulin'] = diabetes['Insulin'].replace(0, medInsulin)
    diabetes['BMI'] = diabetes['BMI'].replace(0, medBMI)

    from sklearn.ensemble import RandomForestClassifier

    # higher estimators value takes longer
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

    total_rows = len(diabetes)
    # training set is 80% of data
    train = diabetes[diabetes.index < .8 * total_rows]
    # test is rest of data
    test = diabetes[diabetes.index > .8 * total_rows]

    # predictor columns
    predictors = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                  'DiabetesPedigreeFunction', 'Age']

    # fit the model
    rf.fit(train[predictors], train["Outcome"])

    preds = rf.predict(test[predictors])

    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(test["Outcome"], preds)
    print(accuracy)
    #accuracy = .7450980392156863
    # how close scores are to target

    combined = pd.DataFrame(dict(actual=test["Outcome"], prediction=preds))

    pd.crosstab(index=combined["actual"], columns=combined["prediction"])

    from sklearn.metrics import precision_score

    precision = precision_score(test["Outcome"], preds)  # 6829268292682927
    print(precision)
    # how close scores are to each other
    from sklearn.metrics import confusion_matrix

    #Plots section
    folder_path = "graphs"

    #check if folder exists, if not, create
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create confusion matrix
    cm = confusion_matrix(test["Outcome"], preds)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of Diabetes Predictor')
    #plt.show()
    plt.savefig(os.path.join(folder_path, "confusionmatrix.png"))

    # Analysis on Relationships

    # Pairplot
    sns.pairplot(diabetes, hue="Outcome", diag_kind='scatter')
    plt.savefig(os.path.join(folder_path, "pairplot.png"))

    # feature importance
    feature_importances = rf.feature_importances_

    # convert to dataframe for readability
    importance_df = pd.DataFrame({'Feature': predictors, 'Importance': feature_importances}).sort_values(
        by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="Blues")
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")

    #plt.show()
    plt.savefig(os.path.join(folder_path, "featureimportance.png"))

    userInput()


#Room for improvement:
#Treat missing values using strategies like iterative imputer.
#Use other models like linear regression, neural network
#Use categorical columns in addition to numerical data
#Remove outliers for each column


