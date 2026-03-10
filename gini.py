# Import library
from sklearn.tree import DecisionTreeClassifier

# Dataset
X = [
    [1, 1],
    [1, 0],
    [0, 1],
    [0, 0]
]

y = ["Pass", "Pass", "Pass", "Fail"]

# Create model using Gini index
model = DecisionTreeClassifier(criterion="gini")

# Train the model
model.fit(X, y)

# Prediction
prediction = model.predict([[0, 0]])

print("Prediction:", prediction)