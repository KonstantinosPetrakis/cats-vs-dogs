"""
This script is a wrapper around the Kaggle API. It loads the Kaggle API key from the .env file and imports the kaggle module.
Just because kaggle thought it would be cool to check the env variables at import time.
"""

import dotenv

dotenv.load_dotenv()
import kaggle