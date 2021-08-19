from flask import Flask, render_template, request, redirect, url_for, Response
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.series import Series
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns

app = Flask(__name__)

# saat dijalankan ini route pertama

page = 1


@app.route('/')
def index():
    return redirect('1')

@app.route('/<string:pg>')
def pageData(pg):
    global page
    try:
        page = int(pg)
        start = 10*page - 10
        finish = 10*page
        df = pd.read_csv('indonesian_movies_clean.csv')

        if(page > 0):
            pg_num = page
            df = df[start:finish]
            return render_template('home.html', df=df)
        else:
            return redirect('1')
        
    except Exception as e:
        return e


@app.route('/search')
def search():
    query = request.args['search']
    try:
        df = pd.read_csv('indonesian_movies_clean.csv')

        df = df[df['title'].str.contains(query)]
        return render_template('home.html', df=df)
    except Exception as e:
        return e

@app.route('/recommendationlist', methods=['GET', 'POST'])
def recommendationlist():
    if request.method == 'POST':
        print (request.form.getlist('like_checkbox'))

    return "done"

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    def create_soup(x):
        return ''.join(x['genre']) + '' + ''.join(x['actors']) + '' + ''.join(x['description']) + ''.join(x['rating'])

    def get_recommendations(title, cosine_sim):
        # Get the index of the movie that matches the title
        idx = indeks[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar movies
    
        return df[['title','year','genre']].iloc[movie_indices]

    df = pd.read_csv('indonesian_movies_clean.csv')
    df['titlelower'] = df['title'].str.lower()

    # df['votes'] = pd.to_numeric(df['votes'].str.replace(',',''))
    judul = str(request.form.get('like_checkbox'))
    judul = judul.lower()

    print(request.form.get('like_checkbox'))

    chars_to_remove = ['$','+',',','[',']',"'",')','(',' ','.','nan']
    # List of column names to clean
    cols_to_clean = df[['genre','rating','actors','description','directors']]
    # Loop for each column in cols_to_clean
    for col in cols_to_clean:
        # Loop for each char in chars_to_remove
        for char in chars_to_remove:
            # Replace the character with an empty string
            df[col] = df[col].apply(lambda x: str(x).replace(char,'').lower())

    df['votes'] = pd.to_numeric(df['votes'].str.replace(',',''))

    df['soup'] = df.apply(create_soup, axis=1) 
    count = CountVectorizer(token_pattern = "[a-zA-Z0-9\-+#]")
    
    count_matrix = count.fit_transform(df['soup'])
    
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    indeks = pd.Series(df.index, index=df['titlelower'])

    df_recom = get_recommendations(judul, cosine_sim)

    # df_recom.to_csv('D:/FinalProject/recom.csv', index=False)

    return render_template('recommendation.html', df=df_recom)

@app.route('/next', methods=['GET', 'POST'])
def next():
    global page
    page += 1
    pg_str = str(page)
    return redirect(pg_str)

@app.route('/prev', methods=['GET', 'POST'])
def prev():
    global page
    page -= 1
    if (page > 0):
        pg_str = str(page)
        return redirect(pg_str)
    else:
        return "Nope"

@app.route('/statistic')
def statistic():
    global page
    df = pd.read_csv('indonesian_movies_clean.csv')
    
    return render_template('statistic.html', df=df)
    

if __name__ == '__main__':
    app.run(debug=True)
