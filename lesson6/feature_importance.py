import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
data = pd.read_csv('combined_credit_data.csv')

# Define features and target
X = data.drop(['approve_credit', 'client_id', 'full_name'], axis=1, errors='ignore')
y = data['approve_credit']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create and train the model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        importance_type='weight',
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

print("Training model...")
model.fit(X_train, y_train)

# Get feature importances
classifier = model.named_steps['classifier']

# Get feature names after preprocessing
feature_names = []

# Get numeric feature names
if len(numeric_features) > 0:
    feature_names.extend(numeric_features)

# Get categorical feature names
if len(categorical_features) > 0:
    ohe = model.named_steps['preprocessor'].named_transformers_['cat']
    try:
        ohe_feature_names = ohe.get_feature_names_out(categorical_features)
        feature_names.extend(ohe_feature_names)
    except Exception as e:
        print(f"Could not get one-hot encoded feature names: {e}")
        feature_names.extend([f"cat_{i}" for i in range(len(categorical_features))])

# Create feature importance dataframe
feature_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': classifier.feature_importances_
}).sort_values('importance', ascending=False)

# Save to CSV
feature_importances.to_csv('feature_importance.csv', index=False)
print("Feature importance saved to 'feature_importance.csv'")

# Plot top 20 features
plt.figure(figsize=(12, 10))
top_n = min(20, len(feature_importances))

# Create the plot
ax = sns.barplot(
    x='importance',
    y='feature',
    data=feature_importances.head(top_n).sort_values('importance', ascending=True),
    palette='viridis'
)

# Customize the plot
plt.title(f'Top {top_n} Most Important Features', fontsize=16, pad=20)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the figure
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Feature importance plot saved as 'feature_importance.png'")
