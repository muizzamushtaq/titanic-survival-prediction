import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = sns.load_dataset("titanic")

# Drop unused columns
df = df.drop(columns=["deck", "embark_town", "alive", "class", "who"])

# Drop rows with missing values
df = df.dropna()

# Convert categorical columns to numeric
df["sex"] = df["sex"].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['embarked'])  # One-hot encode embarked column

# Show cleaned data
print("Cleaned Data:\n", df.head())

# Visual: Survival count
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()

# Split features & label
X = df.drop("survived", axis=1)
y = df["survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.show()
