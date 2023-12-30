from import_modules import *

movies = pd.read_csv('25m\\movies.csv')
ratings = pd.read_csv('25m\\ratings.csv')

def clean_title(title):
    # for the case of "Avengers, The (2012)" or "Bug's Life, A (1998)" or a similar type of format in the Dataset
    # this would rearrange to "The Avengers 2012" or "A Bug's Life 1998"
    match = re.search(r'^(.*), (The|A) \((\d{4})\)$', title)
    if match:
        movie_title = match.group(1)
        year = match.group(2)
        return f"The {movie_title} {year}"
    # handle normally, in which remove only parantheses around the year as that can impact searching
    return re.sub("[^a-zA-Z0-9 ]", "", title)

def rearrange_title(title):
    # for the case of "Avengers, The (2012)" type of format
    # or "Bug's Life, A (1998)"
    # DON'T REMOVE PARANTHESES AROUND YEAR
    match = re.search(r'^(.*), (The|A) \((\d{4})\)$', title)
    if match:
        article = match.group(2)
        movie_title = match.group(1)
        year = match.group(3)
        return f"{article} {movie_title} ({year})"
    return title

def clean_genre(genre):
    if genre == "(no genres listed)":
        return genre
    return re.sub("[^a-zA-Z0-9 ]", ", ", genre)

movies["cleaned_title"] = movies["title"].apply(clean_title)
movies["rearranged_title"] = movies["title"].apply(rearrange_title)
movies["cleaned_genres"] = movies["genres"].apply(clean_genre)

vector = TfidfVectorizer(ngram_range=(1,2))
tfidf = vector.fit_transform(movies["cleaned_title"])

# get unique genres for more robust rec system
def get_genres(dataset, genre_col="genres", delimiter="|"):
    all_genres = dataset[genre_col].str.split(delimiter).explode().str.strip()
    unique_genres = all_genres.unique().tolist()

    unique_genres = [genre for genre in unique_genres if genre != "(no genres listed)"]

    unique_genres = ["Any"] + unique_genres

    return unique_genres

unique_genres = get_genres(movies)

def search(title):
    title = clean_title(title)
    query_vec = vector.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    res = movies.iloc[indices][::-1]
    return res

def find_similar_movies(movie_id, genre=None):
    if genre and genre != "Any":
        movies_genre = movies[movies['genres'].str.contains(genre, case=False, regex=False)]
    else:
        movies_genre = movies

    # find similar users + their recs
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 3.5)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 3.5)]["movieId"]

    # rec only when 10% of users rec that movie
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]

    # find if rec is common with all users
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 3.5)]
    all_users_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    # make a rec "score" from similar users and all users that will display based on searched movie title
    rec_perc = pd.concat([similar_user_recs, all_users_recs], axis=1)
    rec_perc.columns = ["similar", "all"]

    # consider the genre similarity and adjust the score
    rec_perc["score"] = rec_perc["similar"] / rec_perc["all"]
    if genre and genre != "Any":
        genre_bonus = 1 - abs(movies_genre["cleaned_genres"].apply(lambda x: fuzz.ratio(str(x), str(genre))) / 100)
        rec_perc["score"] *= genre_bonus

    rec_perc = rec_perc.sort_values("score", ascending=False)

    # return top 10 recs
    return rec_perc.head(10).merge(movies_genre, left_index=True, right_on="movieId")[["rearranged_title", "cleaned_genres", "score"]]

def main():
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    st.title("Movie Recommendation App")

    if choice == 'Home':
        st.subheader('Home')

        with st.form(key='searchform'):
            n1, n2, n3 = st.columns([3,2,1])
            with n1:
                search_term = st.text_input("Movie Title") 
            with n2:
                genre_term = st.selectbox("Genre", unique_genres)
            with n3:
                st.text("Submit")
                submit = st.form_submit_button(label="Search") 


        if submit:
            if not search_term:
                st.error("Please enter a valid movie title and genre.")
            else:
                if genre_term == "Any":
                    st.success(f"You searched for {search_term}")
                else:
                    st.success(f"You searched for {search_term} and {genre_term}")
                results = search(search_term)
                if not results.empty:
                    movie_id = results.iloc[0]["movieId"]
                    similar_movies = find_similar_movies(movie_id, genre_term)
                    similar_movies_styled = similar_movies.style.format({'rearranged_title': lambda x: f'<a href="https://www.google.com/search?q={x}" target="_blank">{x}</a>'}, escape='html')
                    st.markdown(similar_movies_styled.render(), unsafe_allow_html=True)
                else:
                    st.warning("No search results found. Please try a different movie title.")


    else:
        st.subheader('About')

        st.write

if __name__ == '__main__':
    main()
