import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

# Set style for plots
plt.style.use('ggplot')  # Using 'ggplot' style which is available by default
sns.set_theme(style="whitegrid")
sns.set_palette("viridis")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('house_price.csv')

# 1. Initial Data Exploration
print("\n=== Initial Data Exploration ===")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types and missing values:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())

# 2. Data Cleaning
print("\n=== Data Cleaning ===")
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Check for duplicates
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# 3. Exploratory Data Analysis
print("\n=== Exploratory Data Analysis ===")
# Plot distributions of numerical features
plt.figure(figsize=(15, 10))
num_cols = ['area', 'rooms', 'age', 'distance_to_center', 'price']
for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('numerical_distributions.png')
plt.close()

# Plot categorical features
plt.figure(figsize=(15, 5))
cat_cols = ['city', 'has_garage', 'has_garden']
for i, col in enumerate(cat_cols, 1):
    plt.subplot(1, 3, i)
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('categorical_distributions.png')
plt.close()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# 4. Feature Engineering
print("\n=== Feature Engineering ===")
# Remove area-related features to avoid multicollinearity
# Keep distance_age_interaction as it's not directly related to area
df['distance_age_interaction'] = df['distance_to_center'] * df['age']

# Polynomial feature for distance
df['distance_squared'] = df['distance_to_center'] ** 2

# 4. Combined binary features
df['has_garage_and_garden'] = (df['has_garage'] == 1) & (df['has_garden'] == 1)
df['has_garage_and_garden'] = df['has_garage_and_garden'].astype(int)

# 5. Binning continuous variables
bins = [0, 5, 15, 30, 50, 100]
labels = ['0-5', '6-15', '16-30', '31-50', '51+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)

# 6. Data Preprocessing for Modeling
print("\n=== Data Preprocessing ===")
# Separate features and target
# Drop area and related features that were removed
X = df.drop(['price', 'owner_name'], axis=1)
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define preprocessing for numerical and categorical features
# Removed area-related features to reduce multicollinearity
numeric_features = [
    'rooms', 'age', 'distance_to_center',
    'distance_age_interaction', 'distance_squared'
]

categorical_features = ['city', 'age_group', 'has_garage', 'has_garden', 'has_garage_and_garden']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 6. Model Training and Evaluation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate a model using cross-validation and test set metrics.
    
    Args:
        model: The scikit-learn model to evaluate
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
        ('model', model)
    ])
    
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
        
        # Cross-validation with multiple metrics
        scoring = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error'
        }
        
        cv_results = cross_validate(
            pipeline, X_train, y_train, 
            cv=5, 
            scoring=scoring,
            return_train_score=False
        )
        
        # Calculate mean and std for each metric
        cv_metrics = {}
        for metric in scoring.keys():
            cv_metrics[f'cv_{metric}_mean'] = cv_results[f'test_{metric}'].mean()
            cv_metrics[f'cv_{metric}_std'] = cv_results[f'test_{metric}'].std()
        
        # Calculate feature importance if available
        feature_importance = None
        try:
            if hasattr(model, 'feature_importances_'):
                # For tree-based models
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # For linear models
                feature_importance = np.abs(model.coef_)
        except Exception as e:
            print(f"Could not calculate feature importance: {str(e)}")
        
        return {
            'model': model.__class__.__name__,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            **cv_metrics,
            'feature_importance': feature_importance
        }
        
    except Exception as e:
        print(f"Error evaluating model {model.__class__.__name__}: {str(e)}")
        return {
            'model': model.__class__.__name__,
            'error': str(e)
        }

# Initialize models
models = [
    LinearRegression(),
    Lasso(alpha=0.1, random_state=42),
    Ridge(alpha=1.0, random_state=42),
    ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
]

# Evaluate models
print("\n=== Model Evaluation ===")
results = []
for model in models:
    print(f"\nTraining {model.__class__.__name__}...")
    result = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    if 'error' in result:
        print(f"Error with {model.__class__.__name__}: {result['error']}")
        continue
        
    results.append(result)
    print(f"{result['model']} - "
          f"R2: {result['r2']:.4f}, "
          f"RMSE: {result['rmse']:.2f}, "
          f"MAE: {result['mae']:.2f}, "
          f"MAPE: {result['mape']:.2f}%")

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)

# Display model comparison
print("\n=== Model Comparison ===")
metrics_to_display = ['model', 'r2', 'rmse', 'mae', 'mape', 
                     'cv_r2_mean', 'cv_neg_mse_mean', 'cv_neg_mae_mean']
print(results_df[metrics_to_display].round(4))

# Plot model comparison
plt.figure(figsize=(14, 6))

# R2 Score
plt.subplot(1, 2, 1)
sns.barplot(x='model', y='r2', data=results_df)
plt.title('R2 Score Comparison')
plt.xticks(rotation=45)
plt.ylim(0, 1)

# RMSE
plt.subplot(1, 2, 2)
sns.barplot(x='model', y='rmse', data=results_df)
plt.title('RMSE Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Feature importance analysis
if not results_df.empty:
    best_model_idx = results_df['r2'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'model']
    best_model = models[best_model_idx]
    
    print(f"\n=== Best Model: {best_model_name} ===")
    
    # Create pipeline for feature importance
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Get the preprocessor and model from the pipeline
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    
    # Get feature names after preprocessing
    numeric_features_final = numeric_features  # These stay the same
    
    # Get one-hot encoded feature names
    try:
        onehot_columns = list(preprocessor.named_transformers_['cat'].named_steps['onehot']
                             .get_feature_names_out(categorical_features))
    except Exception as e:
        print(f"Could not get one-hot encoded feature names: {e}")
        onehot_columns = []
    
    all_feature_names = numeric_features_final + onehot_columns
    
    # Get feature importance/coefficients
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_)
    else:
        importances = None
    
    if importances is not None and len(importances) == len(all_feature_names):
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 most important features
        plt.figure(figsize=(12, 8))
        top_n = min(20, len(feature_importance))
        sns.barplot(
            x='importance', 
            y='feature', 
            data=feature_importance.head(top_n),
            palette='viridis'
        )
        plt.title(f'Top {top_n} Most Important Features - {best_model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Print top 10 features
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Save feature importance to CSV
        feature_importance.to_csv('feature_importance.csv', index=False)
    else:
        print("\nCould not calculate feature importance for the best model.")

# Save results to CSV
if not results_df.empty:
    results_df.to_csv('model_results.csv', index=False)
    print("\nModel results saved to 'model_results.csv'")

print("\n=== Analysis Complete ===")
print("Generated files:")
print("- model_comparison.png: Comparison of model performance")
print("- feature_importance.png: Feature importance for the best model")
print("- feature_importance.csv: Detailed feature importance data")
print("- model_results.csv: Complete model evaluation metrics")
