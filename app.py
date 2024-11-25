from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load movie data
column_names = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 
                'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 
                'War', 'Western']
movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='ISO-8859-1', names=column_names, usecols=[0, 1] + list(range(5, 24)))

# Dummy collaborative filtering similarity matrix
np.random.seed(42)
collaborative_matrix = np.random.rand(len(movies), len(movies))
item_similarity_df = pd.DataFrame(collaborative_matrix, index=movies['movie_id'], columns=movies['movie_id'])

# Genre-based similarity matrix
genre_matrix = movies.iloc[:, 2:]  # Use only genre columns
genre_similarity = cosine_similarity(genre_matrix)
genre_similarity_df = pd.DataFrame(genre_similarity, index=movies['movie_id'], columns=movies['movie_id'])

# Enhanced recommendation function
def get_enhanced_recommendations(movie_id, collaborative_matrix, genre_matrix, movies_df, top_n=10, weight_collab=0.5, weight_genre=0.5):
    total_weight = weight_collab + weight_genre
    weight_collab /= total_weight
    weight_genre /= total_weight
    
    collab_scores = collaborative_matrix.loc[movie_id]
    genre_scores = genre_matrix.loc[movie_id]
    
    combined_scores = (weight_collab * collab_scores) + (weight_genre * genre_scores)
    similar_movies = combined_scores.sort_values(ascending=False).index[1:top_n+1]  # Exclude the movie itself
    
    return movies_df.set_index('movie_id').loc[similar_movies, 'movie_title'].tolist()

# Route ke halaman utama
@app.route('/')
def index():
    movie_list = movies[['movie_id', 'movie_title']].to_dict(orient='records')
    return render_template('index.html', movies=movie_list)

# Route untuk hasil rekomendasi
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        movie_id = int(request.form.get('movie_id'))
        recommendations = get_enhanced_recommendations(
            movie_id=movie_id, 
            collaborative_matrix=item_similarity_df, 
            genre_matrix=genre_similarity_df, 
            movies_df=movies, 
            top_n=5, 
            weight_collab=0.7, 
            weight_genre=0.3
        )
        return render_template('result.html', recommendations=recommendations)
    except KeyError:
        return render_template('error.html', message="Movie ID not found.")
    except ValueError:
        return render_template('error.html', message="Invalid Movie ID.")

if __name__ == '__main__':
    print("Flask app is starting...")
    app.run(debug=True)
