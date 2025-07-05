from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Setup
app = Flask(__name__)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing
def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(lemmatizer.lemmatize(w)) for w in tokens]
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv('Online_Courses.csv')
df.dropna(subset=['Title'], inplace=True)
df['processed'] = df['Title'].apply(preprocess)

# Combine fields
df['combined'] = df['processed'] + ' ' + df['Category'].fillna('') + ' ' + df['Platform'].fillna('') + ' ' + df['Duration'].fillna('')

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['combined'])

# KNN model
knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(X)

@app.route('/', methods=['GET', 'POST'])
def index():
    interest_recommendations = []
    course_recommendations = []

    if request.method == 'POST':
        search_mode = request.form.get('submit_button')
        vectorizer_input = ""

        if search_mode == 'interests':
            s1 = request.form.get('search1', '')
            s2 = request.form.get('search2', '')
            s3 = request.form.get('search3', '')
            vectorizer_input = f"{s1} {s2} {s3}"
        elif search_mode == 'course':
            vectorizer_input = request.form.get('course_search', '')

        if vectorizer_input.strip():
            processed = preprocess(vectorizer_input)
            vector_input = vectorizer.transform([processed])
            distances, indices = knn.kneighbors(vector_input)

            recommendations = df.iloc[indices.flatten()].copy()
            recommendations['similarity'] = 1 - distances.flatten()

            # Normalize Rating and ReviewCount if available
            if 'Rating' in recommendations.columns and 'ReviewCount' in recommendations.columns:
                recommendations['Rating'] = pd.to_numeric(recommendations['Rating'], errors='coerce').fillna(0)
                recommendations['ReviewCount'] = pd.to_numeric(recommendations['ReviewCount'], errors='coerce').fillna(0)

                max_reviews = recommendations['ReviewCount'].max()
                if max_reviews > 0:
                    recommendations['review_score'] = recommendations['ReviewCount'] / max_reviews
                else:
                    recommendations['review_score'] = 0

                # Weighted score: adjust the weights as needed
                recommendations['combined_score'] = (
                    0.6 * recommendations['similarity'] +
                    0.2 * recommendations['Rating'] / 5 +  # Assuming rating out of 5
                    0.2 * recommendations['review_score']
                )

                top_recs = recommendations.sort_values(by='combined_score', ascending=False)
            else:
                top_recs = recommendations.sort_values(by='similarity', ascending=False)

            top_recs = top_recs.drop_duplicates(subset=['Platform', 'Category']).head(100)

            if search_mode == 'interests':
                interest_recommendations = top_recs.to_dict(orient='records')
            else:
                course_recommendations = top_recs.to_dict(orient='records')

    return render_template('index.html',
                           interest_recommendations=interest_recommendations,
                           course_recommendations=course_recommendations)

if __name__ == '__main__':
    app.run(debug=True)