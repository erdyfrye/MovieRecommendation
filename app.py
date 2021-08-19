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

@app.route('/1')
def index():
    
    df = pd.read_csv('indonesian_movies_clean.csv')
    df=df[0:20]
    return render_template('home.html', df=df)

@app.route('/<string:pg>')
def pageData(pg):
    global page
    try:
        page = int(pg)
        start = 20*page - 20
        finish = 20*page
        df = pd.read_csv('indonesian_movies_clean.csv')

        if(page > 0):
            pg_num = page
            df = df[start:finish]
            return render_template('rumah.html', df=df)
        else:
            return redirect('2')
        
    except Exception as e:
        return e


@app.route('/search')
def search():
    query = request.args['search']
    try:
        df = pd.read_csv('indonesian_movies_clean.csv')

        df = df[df['titlelower'].str.contains(query)]
        return render_template('home.html', df=df)
    except Exception as e:
        return e
        
@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    def create_soup(x):
        return ''.join(x['genre']) + '' + ''.join(x['actors']) + '' + ''.join(x['description']) + ''.join(x['rating'])

    def get_recommendations(dex, n, cosine_sim):
        # Get the index of the movie that matches the title
        z=[]
        sim_scores=[]
        p=[]
        for i in range(n):
            # Get the pairwsie similarity scores of all movies with that movie
            sim_scores = list(enumerate(cosine_sim[dex[i]]))
            # Sort the movies based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

                
            z = z + sim_scores
        z = pd.DataFrame(z)
        z = z.drop_duplicates(subset=[0])
        z = z.values.tolist()
        z = sorted(z, key=lambda x: x[1], reverse=True)
        p=[]
        for x in z:
            p.append(x[0])
        p = list(map(int, p))
        for c in p:
            if c in dex:
                p.remove(c)

        return df[['title','year','genre']].iloc[p[1:12]]

    listtitle= request.form.getlist('like_checkbox')
    dex = []
    i=0

    df = pd.read_csv('indonesian_movies_clean.csv')
    df['titlelower'] = df['title'].str.lower()

    # df['votes'] = pd.to_numeric(df['votes'].str.replace(',',''))
    judul = str(request.form.get('like_checkbox'))
    judul = judul.lower()

    print(request.form.getlist('like_checkbox'))

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
    
    cosine_simil = cosine_similarity(count_matrix, count_matrix)

    indeks = pd.Series(df.index, index=df['titlelower'])

    listtitle = [each_string.lower() for each_string in listtitle]

    for i in listtitle:
        idx = indeks[i]
        dex.append(idx)

    n = len(listtitle)
    df_recom = get_recommendations(dex, n, cosine_simil)

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
        return redirect('1')

@app.route('/genreplot')
def genreplot():
    global page
    df = pd.read_csv('indonesian_movies_clean.csv')
    img = io.BytesIO()
    fig,ax=plt.subplots(figsize=(10,8))
    sns.countplot(y="genre",data=df).set_title('Total Film berdasarkan Genre')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('visualization.html', plot_url=plot_url)

@app.route('/yearplot')
def yearplot():
    global page
    df = pd.read_csv('indonesian_movies_clean.csv')
    img = io.BytesIO()
    fig,ax=plt.subplots(figsize=(12,10))
    sns.countplot(y="year",data=df).set_title('Total Film berdasarkan Tahun')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('visualization.html', plot_url=plot_url)

@app.route('/ratingplot')
def ratingplot():
    global page
    df = pd.read_csv('indonesian_movies_clean.csv')
    img = io.BytesIO()
    fig,ax=plt.subplots(figsize=(12,10))
    sns.countplot(x="rating",data=df).set_title('Total Film berdasarkan Rating')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('visualization.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
