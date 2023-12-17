import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
import pandas as pd


# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))
df = pd.read_csv('./datasets/movie_metadata.csv')



def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def sort_by_imdb_score(dataset):
    sorted_dataset = dataset.sort_values(by='imdb_score', ascending=False)
    return sorted_dataset


def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())



def get_movie_posters(arr, my_api_key="054feede5a3ca3c1f02bf7fe9e71c761"):
    arr_details_list = []
    base_url = "https://api.themoviedb.org/3/search/movie"
    
    for movie in arr:
        params = {
            'api_key': my_api_key,
            'query': movie
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad requests
            m_data = response.json()

            # Check if the response contains the expected data
            if 'results' in m_data and m_data['results']:
                # Extract title and poster URL
                title = m_data['results'][0]['title']
                poster_path = m_data['results'][0]['poster_path']
                poster_url = f"https://image.tmdb.org/t/p/original{poster_path}" if poster_path else None
            else:
                # Set title and poster to None if data is not available
                title = None
                poster_url = None

            # Append the tuple (title, poster) to the list
            arr_details_list.append((title, poster_url))
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            # Append the tuple (None, None) in case of an error
            arr_details_list.append((None, None))
    
    return arr_details_list

    
app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
     
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
     
    
     
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    
    
    # web scraping to get user reviews from IMDB site
    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup.find_all("div",{"class":"text show-more__control"})

    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # combining reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

    # passing all the data to the html file
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews)

@app.route('/imbd_list')
def index():
    # Sort the dataset by IMDb score
    sorted_dataset = sort_by_imdb_score(df)
    
    # Select the top 10 movies
    top_10 = sorted_dataset.head(10) 
    
    # Extract required columns for each movie
    top_10_info = top_10[['movie_title', 'movie_imdb_link', 'imdb_score']]
    
    # Initialize empty lists to store title and poster details
    titles = []
    posters = []

    # Iterate through each movie to get details
    for _, row in top_10_info.iterrows():
        
        # Fetch title and poster using the movie title from the current row
        details = get_movie_posters([row['movie_title']])[0]
        
        # Ensure that details is not None
        if details:
            title, poster = details

            # Print the title and poster for debugging
            print(title, poster)

            # Append title and poster to the respective lists
            titles.append(title)
            posters.append(poster)

    # Add title and poster details to the DataFrame
    top_10_info['title'] = titles
    top_10_info['poster'] = posters

    # Convert DataFrame to a list of dictionaries
    top_10_list = top_10_info.to_dict(orient='records')

    return render_template('imbd_list.html', top_10_list=top_10_list)


if __name__ == '__main__':
    app.run(debug=True)

