from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess(sentence):
    russian_stopwords = stopwords.words('russian')
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9]", ' ', str(sentence)).split()
    words = [x.lower() for x in text if x.lower() not in russian_stopwords]
    lemma = WordNetLemmatizer()
    lemmatized = [lemma.lemmatize(word, 'v') for word in words]
    return ' '.join(lemmatized)

# Load dataset
df = pd.read_csv("market_comments.csv")

# Clean data
df.dropna(subset=['comment', 'tonality', 'user_id', 'item_id', 'rating'], inplace=True)
df['comment'] = df['comment'].apply(preprocess)

# Train sentiment model
X = df['comment']
y = df['tonality']

sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', MultinomialNB())
])
sentiment_pipeline.fit(X, y)

# Build user-item matrix
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
sparse_matrix = csr_matrix(user_item_matrix.values)
user_similarity_df = pd.DataFrame(cosine_similarity(sparse_matrix), 
                                  index=user_item_matrix.index, 
                                  columns=user_item_matrix.index)

def get_recommendations(user_id, num_recommendations=5):
    if user_id not in user_item_matrix.index:
        return []
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    weighted_ratings = pd.Series(dtype=float)

    for other_user, similarity_score in similar_users.items():
        user_ratings = user_item_matrix.loc[other_user]
        weighted_ratings = weighted_ratings.add(user_ratings * similarity_score, fill_value=0)

    recommendation_scores = weighted_ratings / similar_users.sum()
    already_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = recommendation_scores.drop(index=already_rated, errors='ignore')

    return list(recommendations.sort_values(ascending=False).head(num_recommendations).index)

@app.route('/')
def home():
    return render_template('index.html', comment=None, prediction=None, user_id=None, recommendations=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    comment = request.form['comment']
    cleaned_comment = preprocess(comment)
    prediction = sentiment_pipeline.predict([cleaned_comment])[0]
    return render_template('index.html', comment=comment, prediction=prediction, user_id=None, recommendations=None)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommended_items = get_recommendations(user_id)
    return render_template('index.html', comment=None, prediction=None, user_id=user_id, recommendations=recommended_items)

if __name__ == '__main__':
    app.run(debug=True)
