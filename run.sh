source venv/bin/activate
# Load env vars from .env file
export $(grep -v '^#' .env | xargs)
python app.py
