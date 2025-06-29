from flask import Flask, render_template, request, jsonify
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the saved components (use your actual paths)
with open('model/vectorizer3.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model/tfidf3.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model/movies3.pkl', 'rb') as f:
    movies = pickle.load(f)

def clean_text(text):
    """Clean input text by removing special chars."""
    return re.sub(r"[^a-zA-Z0-9 ]", "", text).lower()

def search(title):
    cleaned = clean_text(title)

    # Find the movie in dataset by exact clean_title match or closest containing
    matched = movies[movies['clean_title'].str.contains(cleaned, case=False, na=False)]

    if matched.empty:
        # No match found, fallback: just search by cleaned title only
        query_text = cleaned
    else:
        # Get genres of matched movie(s) - use first match
        genre = matched.iloc[0]['genres']
        query_text = genre.lower()  # Use genre as query text to prioritize genre similarity

    query_vec = vectorizer.transform([query_text])
    similarity = cosine_similarity(query_vec, tfidf).flatten()

    # Sort top 15 to have room after filtering
    top_indices = similarity.argsort()[-15:][::-1]
    results = movies.iloc[top_indices]

    # Exclude exact matches to the searched title
    results = results[results['clean_title'] != cleaned]

    # Return top 10 after exclusion
    return results.head(10)[["title", "genres"]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    title = data.get('title', '').strip()
    if not title:
        return jsonify([])

    try:
        recommendations = search(title)
        return jsonify(recommendations.to_dict(orient='records'))
    except Exception as e:
        print("Error:", e)
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True)
