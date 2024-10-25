from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Load NLTK resources
nltk.download('punkt_tab')
ps = PorterStemmer()

# Load models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Home route
@app.route('/')
def index():
    return render_template('index.html', result=None)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    message = request.form.get('message')
    if message:
        # 1. Preprocess
        transformed_sms = transform_text(message)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display result
        prediction = "Spam" if result == 1 else "Not Spam"
    else:
        prediction = "No message provided"
    
    return render_template('index.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
