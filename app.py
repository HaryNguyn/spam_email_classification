# ==============================
# app.py - Flask Web App
# ==============================
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Tải mô hình
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']

    # Chuyển text sang vector
    X = vectorizer.transform([email_text])
    pred = model.predict(X)[0]

    result = "🚫 SPAM" if pred == 1 else "✅ HAM (Email bình thường)"
    return render_template('index.html', prediction=result, email_text=email_text)

if __name__ == '__main__':
    app.run(debug=True)
