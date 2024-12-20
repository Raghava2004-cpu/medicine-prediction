import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split , KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

fever_data = pd.read_csv("C:/Users/ragha/Downloads/enhanced_fever_medicine_recommendation.csv")
fever_data.head()
fever_data.drop(columns = ['Previous_Medication'] , inplace = True)
fever_data['Temperature'] = fever_data['Temperature']*(9/5) +32
columns = [ 'Fever_Severity', 'Gender', 'Headache',
       'Body_Ache', 'Fatigue', 'Chronic_Conditions', 'Allergies',
       'Smoking_History', 'Alcohol_Consumption',
       'Physical_Activity', 'Diet_Type', 'Blood_Pressure',
       'Recommended_Medication']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for cols in columns:
   fever_data[cols] = le.fit_transform(fever_data[cols])
   x = fever_data.iloc[: , :-1]
y = fever_data.iloc[: , -1]

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25)
new_model = RandomForestClassifier()
new_model.fit(x_train , y_train)
y_pred = new_model.predict(x_test)
print(accuracy_score(y_test , y_pred))


import pickle
pickle.dump(new_model , open("medicine_prediction.pkl" , 'wb'))
model = pickle.load(open("medicine_prediction.pkl" , 'rb'))

