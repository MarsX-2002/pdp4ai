#!/bin/bash
# Create necessary directories
mkdir -p ~/.streamlit/

# Create streamlit config file
echo "\
[server]\n\nport = $PORT\n\nenableCORS = false\n\nheadless = true\n\n\n[browser]\n\ngatherUsageStats = false\n\nserverAddress = '0.0.0.0'\n" > ~/.streamlit/config.toml

# Copy model file if it doesn't exist in the lesson6 directory
if [ ! -f "lesson6/credit_scoring_model.joblib" ]; then
    # Add commands here to generate the model if needed
    echo "Model file not found. Please ensure credit_scoring_model.joblib is in the lesson6 directory."
fi
