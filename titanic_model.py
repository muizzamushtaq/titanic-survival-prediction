import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Show first 5 rows
print(df.head())

#drop colums we dont need

df =df.drop(columns=["deck","embark_town","alive","class","who"])

#drop row with missing values
df = df.dropna()

# covert sex colum to number
df["sex"] = df["sex"].map({
    'male' : 0 ,
    'female' : 1
})

#convert embrak to number using one hot encoding

df = pd.get_dummies(df,columns=['embarked'])

#show clean data
print(df.head())
print("\ncolums",df.columns)

# Step 1: Split features and label
X = df.drop("survived", axis=1)   # input    
y = df["survived"]                # output
# | Code     | Means                                    |
# | -------- | ---------------------------------------- |
# | `axis=0` | Do something row-wise (🧍 vertical)      |
# | `axis=1` | Do something column-wise (➡️ horizontal) |



# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 4: Predict and check accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", acc)

# # New passenger data
# new_passenger = [[3, 0, 22, 0, 0, 7.25, True, True, 0, 0, 1]]

# columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 
#            'adult_male', 'alone', 'embarked_C', 'embarked_Q', 'embarked_S']

# new_passenger_df = pd.DataFrame([new_passenger[0]], columns=columns)

# # Now predict
# prediction = model.predict(new_passenger_df)

# # Show result
# if prediction[0] == 1:
#     print("Passenger would SURVIVE 🛟")
# else:
#     print("Passenger would NOT survive ❌")
