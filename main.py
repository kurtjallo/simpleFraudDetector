import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("creditcard.csv")

# Check the data
df.head()

# Each row is a transaction, and the 'Class' column indicates fraud (1) or not (0)

x = df.drop("Class", axis=1) # input data (transaction info)
y = df["Class"] # output label (fraud or not)

# Split data into training and testing
# Models learns from training data, abd us tested ib new, unseen data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# Create model
model = LogisticRegression(max_iter=1000) #Logistric Regression predicts probabilities 0, 1

#Train the model
model.fit(x_train, y_train) # The model is learning patterns that separate fraud from normal transactions

# Make predictions
y_pred = model.predict(x_test) # The model is predicting fraud on unseen data

# Evaluate the model
# Precision: How many predicted frauds were actually frauds
# Recall: How many actual frauds were correctly predicted
print(classification_report(y_test, y_pred)) # This shows how well the model performed

# Test a single transaction
# This simulates checking a single credit card transaction for fraud
sample_transaction = x_test.iloc[0].values.reshape(1,-1)
prediction = model.predict(sample_transaction)
print("Predicted class for the sample transaction:", prediction) # Output: 0 (not fraud) or 1 (fraud)







