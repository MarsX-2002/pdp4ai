import streamlit as st
import pandas as pd
import numpy as np
import joblib
from credit_scoring_improved import CreditScoringModel

# Set page config
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="🏦",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model with error handling."""
    try:
        model = joblib.load('credit_scoring_model.joblib')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_credit_approval(model, input_data):
    """Make predictions using the loaded model."""
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        proba = model.predict_proba(input_df)[0]
        prediction = model.predict(input_df)[0]
        
        return {
            'approval_probability': proba[1],  # Probability of approval
            'prediction': 'Approved' if prediction == 1 else 'Rejected',
            'confidence': max(proba) * 100  # Confidence percentage
        }
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def display_metrics_tab(model):
    """Display the evaluation metrics tab with plots."""
    st.header("Model Evaluation Metrics")
    
    if not hasattr(model, 'metrics_') or not model.metrics_:
        st.warning("No evaluation metrics available. Please train the model first.")
        return
    
    metrics = model.metrics_
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    with col2:
        st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
    with col3:
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    with col4:
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
    
    # Display F1 Score and Threshold
    col5, col6 = st.columns(2)
    with col5:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
    with col6:
        st.metric("Optimal Threshold", f"{metrics.get('threshold', 0.5):.4f}")
    
    # Add space
    st.write("")
    
    # Display plots in tabs
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall"])
    
    with tab1:
        st.subheader("Confusion Matrix")
        if hasattr(model, 'y_test') and hasattr(model, 'y_pred'):
            fig = model.plot_confusion_matrix(model.y_test, model.y_pred, metrics.get('threshold', 0.5))
            if fig is not None:
                st.pyplot(fig)
        else:
            st.warning("Test data not available for confusion matrix.")
    
    with tab2:
        st.subheader("ROC Curve")
        if hasattr(model, 'y_test') and hasattr(model, 'y_pred_proba'):
            fig = model.plot_roc_curve(model.y_test, model.y_pred_proba)
            if fig is not None:
                st.pyplot(fig)
        else:
            st.warning("Test data not available for ROC curve.")
    
    with tab3:
        st.subheader("Precision-Recall Curve")
        if hasattr(model, 'y_test') and hasattr(model, 'y_pred_proba'):
            fig = model.plot_precision_recall_curve(model.y_test, model.y_pred_proba)
            if fig is not None:
                st.pyplot(fig)
        else:
            st.warning("Test data not available for precision-recall curve.")

def main():
    st.title("🏦 Credit Scoring Dashboard")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Make Prediction", "Model Info", "Evaluation Metrics"])
    
    if page == "Make Prediction":
        st.header("Credit Approval Prediction")
        
        # Create form for user input
        with st.form("credit_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Personal Information")
                age = st.slider("Age", 18, 100, 30)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                marital_status = st.selectbox(
                    "Marital Status",
                    ["Single", "Married", "Divorced", "Widowed"]
                )
                education_level = st.selectbox(
                    "Education Level",
                    ["High School", "Bachelor's", "Master's", "PhD"]
                )
                employment_years = st.number_input("Years of Employment", 0, 50, 5)
            
            with col2:
                st.subheader("Financial Information")
                income = st.number_input("Annual Income ($)", 0, 500000, 50000, 1000)
                loan_amount = st.number_input("Loan Amount ($)", 1000, 1000000, 20000, 1000)
                monthly_expenses = st.number_input("Monthly Expenses ($)", 0, 10000, 2000, 100)
                savings_balance = st.number_input("Savings Balance ($)", 0, 1000000, 10000, 1000)
                past_due = st.number_input("Number of Past Due Payments (last 12 months)", 0, 12, 0)
            
            region = st.selectbox(
                "Region",
                ["North", "South", "East", "West"]
            )
            
            submitted = st.form_submit_button("Predict")
            
            if submitted:
                # Create input dictionary
                input_data = {
                    'age': age,
                    'gender': gender,
                    'marital_status': marital_status,
                    'education_level': education_level,
                    'employment_years': employment_years,
                    'income': income,
                    'loan_amount': loan_amount,
                    'monthly_expenses': monthly_expenses,
                    'savings_balance': savings_balance,
                    'past_due': past_due,
                    'region': region
                }
                
                # Load model
                model = load_model()
                
                if model is not None:
                    # Make prediction
                    result = predict_credit_approval(model, input_data)
                    
                    if result:
                        # Display results
                        st.subheader("Prediction Results")
                        
                        # Show prediction with color
                        if result['prediction'] == 'Approved':
                            st.success(f"✅ **Prediction:** {result['prediction']}")
                        else:
                            st.error(f"❌ **Prediction:** {result['prediction']}")
                        
                        # Show probability gauge
                        st.metric("Approval Probability", f"{result['approval_probability']*100:.1f}%")
                        
                        # Show explanation
                        with st.expander("What does this mean?"):
                            st.write("""
                            - **Approval Probability**: The model's confidence in the prediction
                            - **Prediction**: Whether the credit application is likely to be approved or rejected
                            - **Confidence**: The model's confidence in the prediction (higher is better)
                            """)
                        
                        # Show feature importance if available
                        try:
                            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None and not model.feature_importances_.empty:
                                st.subheader("Key Factors")
                                st.write("Top factors influencing this decision:")
                                top_features = model.feature_importances_.head(3)
                                if not top_features.empty:
                                    for _, row in top_features.iterrows():
                                        st.write(f"- {row['feature']} (importance: {row['importance']:.3f}")
                                else:
                                    st.warning("No feature importance data available for display.")
                            else:
                                st.warning("Feature importance information is not available for this model.")
                        except Exception as e:
                            st.warning(f"Could not display feature importance: {str(e)}")
                
    elif page == "Model Info":
        st.header("Model Information")
        
        # Load model info if available
        try:
            model = load_model()
            if model is not None:
                st.subheader("Model Performance")
                
                # Display metrics if available
                if hasattr(model, 'metrics_'):
                    metrics = model.metrics_
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2f}")
                    with col2:
                        st.metric("Precision", f"{metrics.get('precision', 0):.2f}")
                    with col3:
                        st.metric("Recall", f"{metrics.get('recall', 0):.2f}")
                    
                    st.metric("F1 Score", f"{metrics.get('f1', 0):.2f}")
                
                # Display feature importance plot if available
                try:
                    if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None and not model.feature_importances_.empty:
                        st.subheader("Feature Importance")
                        fig = model.plot_feature_importance()
                        if fig is not None:  # Only show if we got a figure back
                            st.pyplot(fig)
                        else:
                            st.warning("Could not generate feature importance plot.")
                    else:
                        st.warning("Feature importance data is not available for this model.")
                except Exception as e:
                    st.error(f"Could not display feature importance: {str(e)}")
                    print(f"Error in feature importance plot: {e}")
            
            # Add model training information
            st.subheader("About the Model")
            st.write("""
            This credit scoring model uses XGBoost with the following features:
            - Personal information (age, gender, marital status, etc.)
            - Financial information (income, loan amount, savings, etc.)
            - Credit history (past due payments)
            
            The model is trained to predict the likelihood of credit approval
            based on historical data.
            """)
            
        except Exception as e:
            st.error(f"Error loading model information: {e}")

    elif page == "Evaluation Metrics":
        # Load model for evaluation metrics
        try:
            model = load_model()
            if model is not None:
                display_metrics_tab(model)
            else:
                st.warning("No trained model found. Please train the model first.")
        except Exception as e:
            st.error(f"Error loading model for evaluation: {str(e)}")

if __name__ == "__main__":
    main()
