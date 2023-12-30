# Movie Recommender

Based on dataset from [MovieLens](https://grouplens.org/datasets/movielens/25m/).

Movie Recommendation App built using [StreamLit](https://streamlit.io/) for smooth and simple UI experience. It includes various functions for cleaning and processing movie data, as well as a recommendation system based on user inputted Movie Title and Genre (or lack thereof)


<h3>Libraries Used</h3>

- TfidfVectorizer to vectorize title (make text into numbers that allows for simpler comparing)
- cosine_similarity for searching based on closest number that allows for a more robust searching method

```
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

To run it, call `streamlit run rec-sys.py`

<h2>How does Movie Recommendation work?</h2>
<h3>Find the users who rated a movie (movie_id) and liked it (>3.5/5.0). From that, find the movies they also liked</h3>

```
def find_similar_movies(movie_id, genre=None):
    # find similar users + their recs
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 3.5)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 3.5)]["movieId"]

```

<h3>Narrow down 10% or more of movies users liked that are similar to us</h3>
This is to prevent a return of all movies that other users liked that also liked our movie

```
    # rec only when 10% of users rec that movie
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]
```

<h3>Include all users who have watched movies that were recommended to us.</h3>

Find the percentage of all_users that recommend these movies based on ratings in order to compare with the percentage of similar_users. A larger differential between all_users ratings and similar_users ratings would lead for a larger recommendation

```
    # find if rec is common with all users
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 3.5)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
```

<h3>Generate a score from "similar"/"all"</h3>

If similar and all are similarly rated, they are less likely to be recommended. If not, and similar is much more highly rated compared to all, they are because that is what defines a "niche".

```
    # make a rec "score" from similar users and all users that will display based on searched movie title
    rec_perc = pd.concat([similar_user_recs, all_users_recs], axis=1)
    rec_perc.columns = ["similar", "all"]
    rec_perc["score"] = rec_perc["similar"] / rec_perc["all"]
```

<h3>Return Top 10 Recommendations from this score</h3>

```
    rec_perc = rec_perc.sort_values("score", ascending=False)

    # return top 10 recs
    return rec_perc.head(10).merge(movies_genre, left_index=True, right_on="movieId")
                [["rearranged_title", "cleaned_genres", "score"]]
```

<h3>Including a genre includes an additional factor to recommend by</h3>

If genre is included, movies' scores with the genre is "boosted" using the fuzzy matching ratio from the `fuzzywuzzy` library

```
def find_similar_movies(movie_id, genre=None):
    if genre and genre != "Any":
        movies_genre = movies[movies['genres'].str.contains(genre, case=False, regex=False)]
    else:
        movies_genre = movies

...

    # consider the genre similarity and adjust the score
    rec_perc["score"] = rec_perc["similar"] / rec_perc["all"]
    if genre and genre != "Any":
        genre_bonus = 1 - abs(movies_genre["cleaned_genres"].apply(lambda x: fuzz.ratio(str(x), str(genre))) / 100)
        rec_perc["score"] *= genre_bonus

    rec_perc = rec_perc.sort_values("score", ascending=False)

    # return top 10 recs
    return rec_perc.head(10).merge(movies_genre, left_index=True, right_on="movieId")
                [["rearranged_title", "cleaned_genres", "score"]]

```

<h2>On Landing Page</h2>
<h3>Users are welcome to add a Movie Title (required) and a Genre (optional) to obtain a Movie Recommendation</h3>


![image](https://github.com/ojjc/movie_recommender_1/assets/137390275/bd8d4876-88a9-4662-b99b-2daba6c43160)


<h3>Upon inclusion of a Movie Title, the app generates a list of movies based on a "similarity score"</h3>


![image](https://github.com/ojjc/movie_recommender_1/assets/137390275/5d00508e-fced-4776-9071-8ac5ba94f376)

Titles are hyperlinked to go to a Google Search of the Movie, if the user desired to learn more about the movie


<h3>Including a Genre only recommends movies that include the genre (will need more work to be more robust)</h3>


![image](https://github.com/ojjc/movie_recommender_1/assets/137390275/cb982670-9636-476f-bf5c-785872bf4ee3)


<h3>An empty search will display an error</h3>

![image](https://github.com/ojjc/movie_recommender_1/assets/137390275/9bf75c6f-fafd-40c0-93e1-a323a16397a7)
