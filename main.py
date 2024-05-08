import nltk
import re
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from nltk.sentiment import SentimentIntensityAnalyzer
import uvicorn
import joblib

# Download the required resources
nltk.download('vader_lexicon')

# Load the trained model and vectorizer
model = joblib.load('trained_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Create an instance of the FastAPI
app = FastAPI()

# Create a class to define the request payload structure
class Statement(BaseModel):
    text: str

# Define a route to handle the prediction request
@app.post("/predict")
def predict(statement: Statement):
    text = statement.text.lower()
    features = vectorizer.transform([text])
    predicted_rating = model.predict(features)[0]

    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    if compound_score > 0.2:
        predicted_rating += 0.5
    elif compound_score < -0.2:
        predicted_rating -= 0.5

    predicted_rating = round(predicted_rating)
    predicted_rating = max(1, min(5, predicted_rating))
    return {"predicted_rating": predicted_rating}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    