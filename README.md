# diabetes-predictor
A machine learning predictor for diabetes in python using a random forest classifier.
Uses data from patient records to predict a positive or negative test result of diabetes based on 8 predictive categories.

# Categories:
* Pregnancies - Number of previous pregnancies
* Glucose - Plasma glucose concentration over 2 hours
* BloodPressure - Diastolic blood pressure measured in mm Hg
* SkinThickness - Triceps skin fold thickness measured in mm
* Insulin - Insulin level measurement µU/ml
* BMI - Body mass index
* DiabetesPedigreeFunction - A function that determines type 2 diabetes risk based on family history
* Age - Age in years

# Installation
1. Clone the repository
```
git clone https://github.com/brichoe/diabetes-predictor.git
cd diabetes-predictor
```
2. Install dependencies:
```
pip install -r requirements.txt
```

#Usage
run
```
python predictor.py
```
This will train and test the model, as well as generate supporting graphs in the ```graphs/``` folder.

# Built With:
* [pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [matplotlib](https://matplotlib.org/stable/)
* [seaborn](https://seaborn.pydata.org/)

# Acknowledgements:
Data from https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset/data
