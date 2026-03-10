# Import libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Example dataset
# Features: [Study Hours (>3), Attendance (High)]
X = [
    [1, 1],  # High study, High attendance
    [1, 0],  # High study, Low attendance
    [0, 1],  # Low study, High attendance
    [0, 0]   # Low study, Low attendance
]

# Target variable
y = ["Pass", "Pass", "Pass", "Fail"]

# Create Decision Tree model using ID3 (Entropy)
model = DecisionTreeClassifier(criterion="entropy")

# Train the model
model.fit(X, y)

# Prediction
prediction = model.predict([[0, 0]])

print("Prediction:", prediction)