import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Data Collection
LoanDataframe = pd.read_csv("loan_borowwer_data.csv")
print(LoanDataframe.head(6))

# Data Wrangling
X = LoanDataframe.iloc[:, 2:13]
Y = LoanDataframe["not.fully.paid"]

# Split Data
train_x, test_x, train_y, test_y = train_test_split(
    X, Y, random_state=10, test_size=0.20)

# Model creation and Fitting Model
newModel = RandomForestClassifier()
newModel.fit(train_x, train_y)
print(newModel)

# Data Prediction
prediction = newModel.predict(test_x)

print(prediction)

# Check Accuracy Score
metrics.accuracy_score(prediction, test_y)