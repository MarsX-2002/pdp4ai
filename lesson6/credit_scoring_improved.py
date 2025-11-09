import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, classification_report,
    confusion_matrix, precision_recall_curve, roc_curve, auc
)
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class CreditScoringModel:
    def __init__(self, data_path=None):
        """Initialize the CreditScoringModel with optional data path.
        
        Args:
            data_path (str, optional): Path to the CSV file containing the data.
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.preprocessor = None
        self.feature_importances_ = None
        self.class_distribution = None
        self.metrics_ = {}
        self.feature_names_ = None
        self.target_columns = ['risk_score', 'risk_level', 'approve_credit']
        self.class_distribution = None
        self.feature_importances_ = None
        
    def load_data(self):
        """Load and prepare the dataset."""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.data.shape}")
        return self.data
    
    def preprocess_data(self, target_column='approve_credit'):
        """Preprocess the data for modeling."""
        if target_column not in self.target_columns:
            raise ValueError(f"Target column must be one of {self.target_columns}")
            
        print(f"\nPreprocessing data with target: {target_column}")
        
        # Define columns to drop
        drop_columns = ['client_id', 'full_name']
        
        # Remove target column and other leakage targets from features
        # These are all potential targets or highly correlated with the target
        leakage_targets = ['risk_score', 'risk_level', 'credit_score', target_column]
        
        # Drop unnecessary columns and leakage targets
        self.data = self.data.drop(columns=[col for col in drop_columns if col in self.data.columns])
        
        # Define features and target, ensuring no data leakage
        X = self.data.drop(columns=leakage_targets, errors='ignore')
        
        # Print remaining features for verification
        print("\nSelected features:", X.columns.tolist())
        print("Dropped features due to potential leakage:", leakage_targets)
        y = self.data[target_column]
        
        # Convert target to binary if needed
        if y.dtype == 'object':
            y = y.map({'N': 0, 'Y': 1})  # Assuming 'N'/'Y' for approve_credit
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Get numerical and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop other columns not specified
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def feature_engineering(self):
        """Create new features with proper handling of missing values and log transforms."""
        print("\nPerforming feature engineering...")
        
        def safe_divide(x, y, default=np.nan):
            """Safely divide two arrays, returning default where division is not possible."""
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.where(y != 0, x / y, default)
                return np.where(np.isfinite(result), result, default)
        
        def engineer_features(df):
            df = df.copy()
            
            # 1. Financial Health Metrics
            if all(col in df.columns for col in ['income', 'monthly_expenses']):
                # Disposable income after expenses
                df['disposable_income'] = df['income'] - (df['monthly_expenses'] * 12)
                
                # Savings rate
                df['savings_rate'] = safe_divide(
                    df['savings_balance'], 
                    df['income'], 
                    default=0
                )
                
                # Debt-to-income ratio (if loan_amount represents debt)
                if 'loan_amount' in df.columns:
                    df['debt_to_income'] = safe_divide(
                        df['loan_amount'], 
                        df['income'], 
                        default=0
                    )
            
            # 2. Employment Stability
            if 'employment_years' in df.columns:
                # Categorize employment stability
                bins = [-1, 1, 5, 10, np.inf]
                labels = ['<1', '1-5', '5-10', '10+']
                df['employment_stability'] = pd.cut(
                    df['employment_years'], 
                    bins=bins, 
                    labels=labels
                )
                
                # Binary stability indicator
                df['is_employed_stable'] = (df['employment_years'] >= 2).astype(int)
            
            # 3. Age-related Features
            if 'age' in df.columns:
                # Age groups
                age_bins = [18, 25, 35, 50, 65, np.inf]
                age_labels = ['18-24', '25-34', '35-49', '50-64', '65+']
                df['age_group'] = pd.cut(
                    df['age'], 
                    bins=age_bins, 
                    labels=age_labels
                )
                
                # Interaction: Age and Employment
                if 'employment_years' in df.columns:
                    df['years_employed_per_age'] = safe_divide(
                        df['employment_years'],
                        df['age'] - 18,  # Assuming work starts at 18
                        default=0
                    )
            
            # 4. Credit History Features
            if 'past_due' in df.columns:
                # Past due categories
                df['past_due_category'] = pd.cut(
                    df['past_due'],
                    bins=[-1, 0, 1, 3, 6, np.inf],
                    labels=['0', '1', '2-3', '4-6', '6+'],
                    right=False
                )
                
                # Binary indicator for any past due
                df['has_past_due'] = (df['past_due'] > 0).astype(int)
            
            # 5. Log Transforms for Right-Skewed Features
            for col in ['income', 'loan_amount', 'savings_balance', 'monthly_expenses']:
                if col in df.columns and df[col].min() >= 0:
                    df[f'log1p_{col}'] = np.log1p(df[col])
            
            # 6. Interaction Features
            if all(col in df.columns for col in ['income', 'savings_balance']):
                df['savings_to_income'] = safe_divide(
                    df['savings_balance'],
                    df['income'],
                    default=0
                )
            
            if all(col in df.columns for col in ['loan_amount', 'income']):
                df['loan_to_income'] = safe_divide(
                    df['loan_amount'],
                    df['income'],
                    default=0
                )
            
            # 7. Binning Continuous Variables
            if 'monthly_expenses' in df.columns and 'income' in df.columns:
                expense_ratio = safe_divide(
                    df['monthly_expenses'] * 12,  # Annualize
                    df['income'],
                    default=0
                )
                df['expense_ratio_category'] = pd.cut(
                    expense_ratio,
                    bins=[-0.01, 0.3, 0.5, 0.7, 1.0, np.inf],
                    labels=['<30%', '30-50%', '50-70%', '70-100%', '>100%']
                )
            
            # 8. Missing Value Indicators
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[f'{col}_missing'] = df[col].isnull().astype(int)
            
            # 9. Handle remaining missing values
            for col in df.select_dtypes(include=['number']).columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            # 10. Convert categorical variables to string type to avoid issues with OneHotEncoder
            for col in df.select_dtypes(include=['category']).columns:
                df[col] = df[col].astype(str)
            
            return df
        
        # Apply feature engineering to both train and test sets
        self.X_train = engineer_features(self.X_train)
        self.X_test = engineer_features(self.X_test)
        
        return self.X_train, self.X_test
    
    def train_model(self, tune_hyperparams=False, cv_folds=5):
        """Train the XGBoost model with cross-validation and optional hyperparameter tuning."""
        print(f"Training XGBoost model with {cv_folds}-fold cross-validation...")
        
        # Calculate scale_pos_weight for imbalanced classes
        n_pos = (self.y_train == 1).sum()  # Majority class (approvals)
        n_neg = (self.y_train == 0).sum()  # Minority class (rejections)
        scale_pos_weight = n_pos / max(n_neg, 1)  # Avoid division by zero
        
        # Store class distribution info
        self.class_distribution = {
            'positive': n_pos,
            'negative': n_neg,
            'total': len(self.y_train),
            'positive_ratio': n_pos / len(self.y_train) if len(self.y_train) > 0 else 0,
            'negative_ratio': n_neg / len(self.y_train) if len(self.y_train) > 0 else 0,
            'imbalance_ratio': n_neg / n_pos if n_pos > 0 else float('inf')
        }
        
        print(f"Class distribution: {n_neg} negative (rejections), {n_pos} positive (approvals)")
        print(f"Imbalance ratio (negative/positive): {(n_neg/n_pos):.2f} to 1" if n_pos > 0 else "No positive samples")
        print(f"Using scale_pos_weight = {scale_pos_weight:.2f} to upweight minority class (rejections)")
        
        # Define base XGBoost parameters with class weights
        xgb_params = {
            'n_estimators': 200,  # Fixed number of estimators since early stopping might not work
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False,
            'scale_pos_weight': scale_pos_weight,
            'min_child_weight': 3,
            'gamma': 0.2,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'tree_method': 'hist',
            'n_jobs': -1,
            'objective': 'binary:logistic',
            'importance_type': 'weight',
            'verbosity': 1,
            'eval_metric': 'aucpr'  # Moved here as it's a model parameter
        }
        
        # Split data for early stopping
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )
        
        # Create and train the model with early stopping
        print("Training with early stopping on validation set...")
        model = XGBClassifier(**xgb_params)
        
        # Create preprocessor
        preprocessor = self.preprocessor
        
        # Preprocess the data
        X_train_processed = preprocessor.fit_transform(X_train_split)
        X_val_processed = preprocessor.transform(X_val)
        
        # Train with early stopping
        try:
            # First try with early stopping
            model.fit(
                X_train_processed, y_train_split,
                eval_set=[(X_val_processed, y_val)],
                early_stopping_rounds=10,
                verbose=10,
                eval_metric='aucpr'
            )
        except (TypeError, Exception) as e:
            print(f"Warning: Early stopping not supported with this XGBoost version. Error: {e}")
            print("Falling back to training without early stopping...")
            # If early stopping fails, train without it
            model.fit(
                X_train_processed, y_train_split,
                verbose=10
            )
        
        # Create final pipeline
        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Save feature importances if available
        try:
            importances = model.feature_importances_
            feature_names = preprocessor.get_feature_names_out()
            
            # Create and save feature importance dataframe
            self.feature_importances_ = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Save to CSV
            self.feature_importances_.to_csv('feature_importance.csv', index=False)
            print("Feature importance saved to 'feature_importance.csv'")
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(
                x='importance',
                y='feature',
                data=self.feature_importances_.head(20).sort_values('importance', ascending=True),
                palette='viridis'
            )
            plt.title('Top 20 Most Important Features')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
            print("Feature importance plot saved as 'feature_importance.png'")
            
        except Exception as e:
            print(f"Could not generate feature importance: {e}")
        
        return self.model
    
    def _extract_feature_importances(self):
        """Extract and store feature importances from the trained model."""
        try:
            # Get the classifier from the pipeline
            if hasattr(self.model, 'named_steps'):
                classifier = self.model.named_steps.get('classifier')
            else:
                classifier = self.model
            
            if hasattr(classifier, 'feature_importances_'):
                # Get feature names after preprocessing
                if hasattr(self, 'preprocessor') and hasattr(self.preprocessor, 'get_feature_names_out'):
                    feature_names = self.preprocessor.get_feature_names_out()
                else:
                    feature_names = [f'feature_{i}' for i in range(len(classifier.feature_importances_))]
                
                # Store feature importances
                self.feature_importances_ = pd.DataFrame({
                    'feature': feature_names,
                    'importance': classifier.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save to CSV
                self.feature_importances_.to_csv('feature_importance.csv', index=False)
                print("Feature importance saved to 'feature_importance.csv'")
                
        except Exception as e:
            print(f"Error extracting feature importances: {e}")
            self.feature_importances_ = None
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            
        Returns:
            matplotlib.figure.Figure: The figure object containing the plot
        """
        if not hasattr(self, 'feature_importances_') or self.feature_importances_ is None:
            self._extract_feature_importances()
            
        if not hasattr(self, 'feature_importances_') or self.feature_importances_ is None:
            print("No feature importances available to plot.")
            return None
            
        try:
            # Limit to top N features
            df_plot = self.feature_importances_.head(top_n).sort_values('importance', ascending=True)
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            ax = sns.barplot(
                x='importance',
                y='feature',
                data=df_plot,
                palette='viridis'
            )
            
            # Customize the plot
            plt.title(f'Top {len(df_plot)} Most Important Features', fontsize=16, pad=20)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save the figure
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Feature importance plot saved as 'feature_importance.png'")
            return ax.figure
            
        except Exception as e:
            print(f"Error plotting feature importances: {e}")
            return None
    
    def plot_confusion_matrix(self, y_true, y_pred, threshold):
        """Plot a confusion matrix with proper labels and formatting.
        
        Returns:
            matplotlib.figure.Figure: The figure object containing the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Rejected', 'Approved'],
                   yticklabels=['Rejected', 'Approved'],
                   ax=ax)
        ax.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve with AUC score.
        
        Returns:
            matplotlib.figure.Figure: The figure object containing the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True)
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba):
        """Plot precision-recall curve with average precision score.
        
        Returns:
            matplotlib.figure.Figure: The figure object containing the plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, lw=2, color='navy',
                label=f'Precision-Recall (AP = {avg_precision:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.legend(loc='best')
        ax.grid(True)
        plt.tight_layout()
        return fig
    
    def find_optimal_threshold(self, y_true, y_pred_proba, method='f1'):
        """Find the optimal threshold based on different criteria.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            method: Method to find threshold ('f1', 'precision_recall', 'youden')
            
        Returns:
            float: Optimal threshold
        """
        if method == 'f1':
            # Find threshold that maximizes F1 score
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx]
        
        elif method == 'precision_recall':
            # Find threshold that maximizes the f1 score from precision-recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx]
        
        elif method == 'youden':
            # Find threshold that maximizes Youden's J statistic (sensitivity + specificity - 1)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            return thresholds[optimal_idx]
        
        else:
            return 0.5  # Default threshold
    
    def evaluate_model(self, threshold_method='f1'):
        """Evaluate the model on the test set.
        
        Args:
            threshold_method (str): Method to determine optimal threshold ('f1' or 'youden').
            
        Returns:
            dict: Dictionary of evaluation metrics.
        """
        if self.model is None:
            raise ValueError("No model available for evaluation.")
            
        print("\nEvaluating model...")
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Find optimal threshold
        threshold = self.find_optimal_threshold(self.y_test, y_pred_proba, method=threshold_method)
        
        # Make predictions using the threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Store test predictions for later use
        self.y_test = self.y_test  # Store test labels
        self.y_pred = y_pred  # Store predictions
        self.y_pred_proba = y_pred_proba  # Store prediction probabilities
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'average_precision': average_precision_score(self.y_test, y_pred_proba),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred, zero_division=0),
            'threshold': threshold
        }
        
        # Print metrics
        print("\nModel Performance Metrics:")
        print(f"- Accuracy: {metrics['accuracy']:.4f}")
        print(f"- ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"- Average Precision: {metrics['average_precision']:.4f}")
        print(f"- Precision: {metrics['precision']:.4f}")
        print(f"- Recall: {metrics['recall']:.4f}")
        print(f"- F1 Score: {metrics['f1_score']:.4f}")
        print(f"- Decision Threshold: {metrics['threshold']:.4f}")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(self.y_test, y_pred, threshold)
        
        # Plot ROC curve
        self.plot_roc_curve(self.y_test, y_pred_proba)
        
        # Plot precision-recall curve
        self.plot_precision_recall_curve(self.y_test, y_pred_proba)
        
        # Plot feature importance
        self.plot_feature_importance()
        
        # Store metrics in the model instance for later reference
        self.metrics_ = metrics
        self.optimal_threshold_ = threshold
        
        # Ensure feature importances are set and saved with the model
        self._extract_feature_importances()
        
        # Save the model with all attributes
        self.save_model('credit_scoring_model.joblib')
        
        return metrics
    
    def predict(self, X, threshold=None):
        """Predict class labels for X.
        
        Args:
            X (DataFrame or array-like): Input features.
            threshold (float, optional): Decision threshold. If None, uses the optimal threshold.
            
        Returns:
            ndarray: Predicted class labels.
        """
        if threshold is None:
            # Use the optimal threshold if available, otherwise use 0.5
            threshold = getattr(self, 'optimal_threshold_', 0.5)
            
        proba = self.predict_proba(X)
        if proba is not None:
            return (proba[:, 1] >= threshold).astype(int)
        return None
    
    def predict_proba(self, X):
        """Predict probabilities for X.
        
        Args:
            X (DataFrame or array-like): Input features.
            
        Returns:
            ndarray: Predicted probabilities.
        """
        if self.model is not None:
            return self.model.predict_proba(X)
        return None
    
    def save_model(self, filepath='credit_scoring_model.joblib'):
        """Save the trained model to a file.
        
        Args:
            filepath (str): Path where to save the model.
        """
        if self.model is not None:
            # Store feature names if available
            if hasattr(self, 'feature_names_'):
                self.feature_names_ = self.X_train.columns.tolist()
            
            # Save the model
            joblib.dump(self, filepath)
            print(f"Model saved to {filepath}")
            return True
        else:
            print("No model to save. Please train the model first.")
            return False
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        # Create a new instance
        instance = cls.__new__(cls)
        
        # Set attributes from saved model
        instance.model = model_data['model']
        instance.preprocessor = model_data['preprocessor']
        instance.feature_importances_ = model_data['feature_importances']
        instance.class_distribution = model_data['class_distribution']
        
        return instance


def main():
    # Initialize the model
    model = CreditScoringModel('combined_credit_data.csv')
    
    # Load and preprocess data
    data = model.load_data()
    
    # Print basic info
    print("\nDataset Info:")
    print(data.info())
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(data.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    # Preprocess data with 'approve_credit' as target
    print("\nPreprocessing data...")
    model.preprocess_data(target_column='approve_credit')
    
    # Perform feature engineering
    print("\nPerforming feature engineering...")
    model.feature_engineering()
    
    # Train the model (set tune_hyperparams=True for hyperparameter tuning)
    print("\nTraining model...")
    model.train_model(tune_hyperparams=False)
    
    # Evaluate the model
    print("\nEvaluating model...")
    metrics = model.evaluate_model(threshold_method='f1')
    
    # Save the trained model
    model.save_model('credit_scoring_model.joblib')
    
    print("\nCredit scoring model training completed successfully!")


if __name__ == "__main__":
    main()
