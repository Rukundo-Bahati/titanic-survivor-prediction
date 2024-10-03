import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('train.csv')

data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

X = data.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)  # drop non-feature columns
y = data['Survived']

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Build a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
