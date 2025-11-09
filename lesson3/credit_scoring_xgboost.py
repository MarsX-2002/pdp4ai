import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import xgboost as xgb
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('scoring_raw.csv')
print(f"Dataset shape: {df.shape}")

# Display basic info
print("\n=== Dataset Info ===")
df.info()

# Display basic statistics
print("\n=== Basic Statistics ===")
print(df.describe())

# Check for missing values
print("\n=== Missing Values ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Check class distribution
print("\n=== Class Distribution ===")
print(df['default_flag'].value_counts(normalize=True))

# Visualize the target variable
plt.figure(figsize=(8, 5))
sns.countplot(x='default_flag', data=df)
plt.title('Class Distribution (0: Non-Default, 1: Default)')
plt.savefig('class_distribution.png')
plt.close()

# Select numerical and categorical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove target variable from numerical features
if 'default_flag' in numerical_features:
    numerical_features.remove('default_flag')

print(f"\nNumerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Plot distributions of numerical features
print("\nPlotting numerical feature distributions...")
plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_features[:12], 1):  # Plot first 12 numerical features
    plt.subplot(3, 4, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('numerical_distributions.png')
plt.close()

# Plot correlation matrix
print("Plotting correlation matrix...")
plt.figure(figsize=(16, 12))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Data Preprocessing
print("\n=== Data Preprocessing ===")

# Handle missing values
for col in numerical_features:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical variables
label_encoders = {}
for col in categorical_features:
    if df[col].nunique() <= 10:  # Label encode for features with few categories
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    else:  # One-hot encode for high cardinality features
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Split the data
X = df.drop('default_flag', axis=1)
y = df['default_flag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost Model Training
print("\n=== XGBoost Model Training ===")

# Define the model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Performing grid search...")
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_xgb = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

# Model Evaluation
print("\n=== Model Evaluation ===")

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Print classification reports
    print("\nTraining Classification Report:")
    print(classification_report(y_train, y_train_pred))
    
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Calculate and print ROC AUC
    train_auc = roc_auc_score(y_train, y_train_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)
    print(f"\nTrain ROC AUC: {train_auc:.4f}")
    print(f"Test ROC AUC: {test_auc:.4f}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Plot feature importance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [X.columns[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Evaluate the best model
evaluate_model(best_xgb, X_train_scaled, X_test_scaled, y_train, y_test)

# Save the model
import joblib
joblib.dump(best_xgb, 'xgboost_credit_scoring_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("\n=== Model Training Complete ===")
print("Model and preprocessing objects have been saved to disk.")
