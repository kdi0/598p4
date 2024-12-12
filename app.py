import pandas as pd
import joblib
import numpy as np
import os


movies = pd.read_csv('movies.dat', sep='::', engine = 'python',
                     encoding="ISO-8859-1", header = None)
movies.columns = ['MovieID', 'Title', 'Genres']

# read the movie and user id
ratings_pivot_columns=joblib.load('ratings_pivot_columns.pkl')
ratings_pivot_index=joblib.load('ratings_pivot_index.pkl')

# read rankings
rankings = joblib.load('rankings.pkl')

# read sparse and convert to S_top30_masked
S_top30_masked_df=joblib.load('S_top30_masked_df.pkl')
S_top30_masked = S_top30_masked_df.sparse.to_dense().to_numpy()

def myIBCF(newuser):
    assert newuser.shape == (3706,1)
    N = newuser.shape[0]
    v = newuser.reshape(N,)
    user_movie_list = np.argwhere(~np.isnan(v)).reshape(-1).tolist()
    
    C = np.nan_to_num(S_top30_masked) @ np.nan_to_num(newuser)
    D = np.nan_to_num(S_top30_masked) @ np.nan_to_num(newuser*0+1)
    ratings_pred = C/D
    ratings_pred[np.where(~np.isnan(newuser))]=np.nan
    ratings_pred = ratings_pred.reshape(N,)
    
    count_valid = np.count_nonzero(~np.isnan(ratings_pred))
    #print(count_valid)
    top10_idx = np.argsort(np.nan_to_num(ratings_pred, nan=-np.inf))[::-1][:10]
    recommendation = np.array(ratings_pivot_columns[top10_idx])
    rankings_excluded = rankings['MovieID'][~rankings['MovieID'].isin(user_movie_list)]
    if count_valid < 10:
          # can read rankings from file
          # must not be reviewed
          recommendation[-(10-count_valid):]=rankings_excluded.iloc[:10-count_valid].to_numpy()
    return recommendation

import streamlit as st
from PIL import Image
#import requests
#from io import BytesIO

st.set_page_config(layout="wide")
movie_ids = ratings_pivot_columns
movies_per_page = 8
movies_per_row = 8
total_movies = len(movie_ids)
total_pages = total_movies // movies_per_page

def get_movie_thumbnail(movie_id):
    src = 'MovieImages/'+str(movie_id)+'.jpg'
    return src

if 'ratings' not in st.session_state:
    st.session_state['ratings'] = {movie_id: np.nan for movie_id in movie_ids}

#if 'current_page' not in st.session_state:
#    st.session_state['current_page'] = 0 

if 'current_page' not in st.session_state:
    st.session_state.current_page = 0
    
st.title("Movie Recommender System - 651627872")
st.write("Please rate below movies 1-5 star(s)，Leave blank for those you don't want to rate.")

# page function to show movies

def show_movies(page):
    start_idx = page * movies_per_page
    end_idx = min(start_idx + movies_per_page, total_movies)
    current_movies = movie_ids[start_idx:end_idx]

    for i in range(0, len(current_movies), movies_per_row):
        cols = st.columns(movies_per_row)  # 4 movies per row
        for col, movie_id in zip(cols, current_movies[i:i+movies_per_row]):

            #thumbnail_url = get_movie_thumbnail(movie_id)
            #response = requests.get(thumbnail_url)
            #image = Image.open(BytesIO(response.content))
            title = movies.loc[movies['MovieID']==movie_id].iloc[0]['Title']
            thumbnail_path = get_movie_thumbnail(movie_id)
            
            if os.path.exists(thumbnail_path): 
                image = Image.open(thumbnail_path)
                col.image(image, width=100, caption=f"m{movie_id}:{title}")
            else:
                col.write(f"Movie {movie_id}: image not found")
                
            # stars selector，np.nan by default for no ratings
            rating = col.radio(f"Rate (m{movie_id})", [1, 2, 3, 4, 5], index=None, key=f"movie_{movie_id}")
            #rating = col.radio(' ', [np.nan, 1, 2, 3, 4, 5], index=0, key=f"movie_{movie_id}")
            # save ratings
            if rating is None:
                rating = np.nan
            st.session_state['ratings'][movie_id] = rating

# show current page movies
show_movies(st.session_state['current_page'])

# preload next page function
def preload_next_page(page):
    next_page = page + 1
    if next_page < total_pages:
        start_idx = next_page * movies_per_page
        end_idx = min(start_idx + movies_per_page, total_movies)
        next_movies = movie_ids[start_idx:end_idx]

        for movie_id in next_movies:
            #thumbnail_url = get_movie_thumbnail(movie_id)
            #requests.get(thumbnail_url) 
            # load from local path
            thumbnail_path = get_movie_thumbnail(movie_id)
            if os.path.exists(thumbnail_path):
                # cache
                Image.open(thumbnail_path)

# preload next page
preload_next_page(st.session_state['current_page'])

# show navigate button
col1, col2 = st.columns([1, 4])

#st.write(f"Page: {st.session_state.current_page}")


# next page
def nextpage():
    if st.session_state.current_page  < total_pages - 1:
            st.session_state.current_page += 1

with col1:
    st.button("Next batch", on_click=nextpage)


    nextpage="""
    if st.button("Last batch"):
        if st.session_state.current_page > 0:
            st.session_state.current_page -= 1  
            st.query_params(rerun="true")
    """
# submit button
if st.button("Submit Ratings"):
    # convert to array
    ratings_array = np.array([st.session_state['ratings'][movie_id] for movie_id in movie_ids])

    # recommendation
    recommended_movie_ids = myIBCF(ratings_array.reshape(-1,1))

    # refresh and show recommendation
    st.write("Recommended Movies：")
    cols = st.columns(10)  # show 10 movies
    for i in range(0, len(recommended_movie_ids), 10):
        cols = st.columns(10)  # 5 movies per row
        for col, movie_id in zip(cols, recommended_movie_ids[i:i+10]):
            title = movies.loc[movies['MovieID'] == movie_id].iloc[0]['Title']
            thumbnail_path = get_movie_thumbnail(movie_id)

            if os.path.exists(thumbnail_path): 
                image = Image.open(thumbnail_path)
                col.image(image, width=100, caption=f"m{movie_id}: {title}")
            else:
                col.write(f"Movie {movie_id}: image not found")
