import sys
import os

# Add the lesson6 directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your dashboard
from lesson6.credit_scoring_dashboard import main

if __name__ == "__main__":
    main()
