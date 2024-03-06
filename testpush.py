#Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
data1 = {
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'target_column': [0, 1, 0, 1, 0]
}
data = pd.DataFrame(data1)   

X = data.drop('target_column', axis=1)
print(X)
y = data['target_column']
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict([[18, 9000]])
print(prediction)

#Streamlit application
st.title("Machine Learning Classifier")

#Input fields for features
feature1 = st.number_input("Feature 1", value=0)
feature2 = st.number_input("Feature 2", value=0)
#Add more input fields for other features...
#Validation button
if st.button("Predict"):
    # Make predictions
    prediction = clf.predict([[feature1, feature2]])[0]
    st.write(f"Predicted class: {prediction}")
