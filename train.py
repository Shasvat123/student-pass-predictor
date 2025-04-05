from sklearn.linear_model import LogisticRegression
import joblib

# Sample training data
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
     [1.5], [2.5], [3.5], [4.5], [5.5], [6.5], [7.5], [8.5], [9.5], [10.5],
     [0.5], [0.8], [1.2], [2.2], [6.8]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
     0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 0, 1]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save to model.pkl
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")
