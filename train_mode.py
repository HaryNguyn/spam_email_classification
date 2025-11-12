import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# 1. Lấy dữ liệu (từ UCI)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/SMSSpamCollection"
data = pd.read_csv(r"smsspamcollection\SMSSpamCollection",
                   sep='\t', names=['label', 'message'])
# 2. Tiền xử lý cơ bản
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# 3. Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label_num'], test_size=0.2, random_state=42
)

# 4. Vector hóa text
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Huấn luyện mô hình
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Đánh giá nhanh
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 7. Lưu mô hình & vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✅ Mô hình và vectorizer đã được lưu!")
