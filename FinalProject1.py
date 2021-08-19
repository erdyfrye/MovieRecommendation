
from math import sqrt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt



df = pd.read_csv('indonesian_movies.csv')

#Replace NaN with an empty string
df['description'] = df['description'].fillna('')
df['directors'] = df['directors'].fillna('')


df = df.drop(['languages','runtime','votes','users_rating'], axis = 1)


print(df['genre'].nunique(), 'unique genres:')
print(df['genre'].unique())

print(df['rating'].nunique(), 'unique ratings')
print(df['rating'].unique())


df['rating'] = df['rating'].fillna('Unrated')
df['rating'] = df['rating'].replace({
    "Not Rated": "Unrated",
    "R": "13+",
    "PG-13": "13+",
    "TV-14": "13+",
    "TV-MA": "17+",
    "D": "17+",
    "21+": "17+"
})
print(df['rating'].nunique(), 'unique ratings')
print(df['rating'].unique())




print(df[df["genre"].isnull()])




df['genre'] = df['genre'].fillna('Undefine')



df.drop_duplicates(subset=['title'])





df['titlelower'] = df['title'].str.lower()



chars_to_remove = ['$','+',',','[',']',"'",')','(',' ','.','nan']
# List of column names to clean
cols_to_clean = df[['genre','rating','actors','description','directors']]
# Loop for each column in cols_to_clean
for col in cols_to_clean:
    # Loop for each char in chars_to_remove
    for char in chars_to_remove:
        # Replace the character with an empty string
        df[col] = df[col].apply(lambda x: x.replace(char,'')).str.lower()




def create_soup(x):
    return ''.join(x['genre'])+ ''.join(x['description'])+''.join(x['actors'])+''.join(x['rating'])





# Create a new soup feature
df['soup'] = df.apply(create_soup, axis=1)


# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(token_pattern = "[a-zA-Z0-9]")
count_matrix = count.fit_transform(df['soup'])



# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix, count_matrix)



#Construct a reverse map of indices and movie titles
indeks = pd.Series(df.index, index=df['titlelower'])



# Function that takes in movie title as input and outputs most similar movies

def get_recommendations(dex, n, cosine_sim=cosine_sim):
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
    print(df[['title','year','genre']].iloc[p[1:12]])
    return 


listtitle=[]
dex = []
i=0
while i <= len(df):
    judul = input('Masukkan Judul: ')
    listtitle.append(judul)
    i=+1
    if judul == '':
        break

del listtitle[-1]
listtitle = [each_string.lower() for each_string in listtitle]

for i in listtitle:
    idx = indeks[i]
    dex.append(idx)

get_recommendations(dex,n=len(listtitle))




