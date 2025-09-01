import pandas as pd
from sklearn.model_selection import train_test_split

# Titanic dataset load
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Features & Target
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]   # features
y = df['Survived']
                                 # target
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Columns
numeric_features = ['Age', 'Fare']
categorical_features = ['Sex', 'Embarked', 'Pclass']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

from sklearn.linear_model import LogisticRegression

# Full pipeline = Preprocessing + Model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=2000))
])

from sklearn.model_selection import GridSearchCV

# Hyperparameter grid
param_grid = {
    'model__C': [0.01, 0.1, 1, 10],        # Regularization strength
    'model__penalty': ['l1', 'l2'],        # Type of regularization
    'model__solver': ['liblinear']         # Solver that supports l1/l2
}

# GridSearchCV
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)

from sklearn.model_selection import cross_val_score, KFold

# Outer CV
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV GridSearchCV
grid_nested = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')

# Nested CV evaluation
nested_scores = cross_val_score(grid_nested, X, y, cv=outer_cv)

print("Nested CV Accuracy Scores:", nested_scores)
print("Mean Accuracy:", nested_scores.mean())

import joblib

# Train final model with best params
best_model = grid.best_estimator_
print (best_model)
# Save model
joblib.dump(best_model, "titanic_pipeline.pkl")

# Load model
loaded_model = joblib.load("titanic_pipeline.pkl")

# Predict with loaded model
sample = pd.DataFrame({
    'Pclass': [1],
    'Sex': ['female'],
    'Age': [22],
    'Fare': [7.25],
    'Embarked': ['S']
})

print("Prediction (Loaded Model):", loaded_model.predict(sample))
