import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import ipywidgets as widgets
from IPython.display import display

# Step 1: Load dataset
data = pd.read_csv("cirrhosis.csv")

# Step 2: Data Preprocessing
# Fill missing values
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    data[column].fillna(data[column].median(), inplace=True)
for column in data.select_dtypes(include=['object']).columns:
    data[column].fillna(data[column].mode()[0], inplace=True)

# Encode categorical variables
le_dict = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    le_dict[column] = le

# Feature Selection
X = data.drop(columns=['Status', 'ID'])
y = data['Status']

# Handle Class Imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# Prediction Widget
def predict_status(n_days, drug, age, sex, ascites, hepatomegaly, spiders, edema, bilirubin, cholesterol, albumin, copper, alk_phos, sgot, tryglicerides, platelets, prothrombin, stage):
    new_data_sample = pd.DataFrame([{
        'N_Days': n_days,
        'Drug': drug,
        'Age': age,
        'Sex': sex,
        'Ascites': ascites,
        'Hepatomegaly': hepatomegaly,
        'Spiders': spiders,
        'Edema': edema,
        'Bilirubin': bilirubin,
        'Cholesterol': cholesterol,
        'Albumin': albumin,
        'Copper': copper,
        'Alk_Phos': alk_phos,
        'SGOT': sgot,
        'Tryglicerides': tryglicerides,
        'Platelets': platelets,
        'Prothrombin': prothrombin,
        'Stage': stage
    }])

    new_data_sample_scaled = scaler.transform(new_data_sample)
    prediction_sample = rf.predict(new_data_sample_scaled)
    print("Predicted Status for New Sample:", prediction_sample[0])

# Widgets
n_days = widgets.IntText(description='N_Days:', value=1925)
drug = widgets.IntSlider(description='Drug:', min=0, max=1, value=0)
age = widgets.IntText(description='Age:', value=50)
sex = widgets.IntSlider(description='Sex:', min=0, max=1, value=0)
ascites = widgets.IntSlider(description='Ascites:', min=0, max=1, value=0)
hepatomegaly = widgets.IntSlider(description='Hepatomegaly:', min=0, max=1, value=1)
spiders = widgets.IntSlider(description='Spiders:', min=0, max=1, value=1)
edema = widgets.IntSlider(description='Edema:', min=0, max=1, value=1)
bilirubin = widgets.FloatText(description='Bilirubin:', value=1.8)
cholesterol = widgets.FloatText(description='Cholesterol:', value=244.0)
albumin = widgets.FloatText(description='Albumin:', value=2.54)
copper = widgets.FloatText(description='Copper:', value=64.0)
alk_phos = widgets.FloatText(description='Alk_Phos:', value=6121.0)
sgot = widgets.FloatText(description='SGOT:', value=60.63)
tryglicerides = widgets.FloatText(description='Tryglicerides:', value=92.0)
platelets = widgets.FloatText(description='Platelets:', value=183.0)
prothrombin = widgets.FloatText(description='Prothrombin:', value=10.9)
stage = widgets.FloatText(description='Stage:', value=4.0)

predict_button = widgets.Button(description='Predict')
predict_button.on_click(lambda b: predict_status(n_days.value, drug.value, age.value, sex.value, ascites.value, hepatomegaly.value, spiders.value, edema.value, bilirubin.value, cholesterol.value, albumin.value, copper.value, alk_phos.value, sgot.value, tryglicerides.value, platelets.value, prothrombin.value, stage.value))

# Display Widgets
display(n_days, drug, age, sex, ascites, hepatomegaly, spiders, edema, bilirubin, cholesterol, albumin, copper, alk_phos, sgot, tryglicerides, platelets, prothrombin, stage, predict_button)
