import pandas as pd
import streamlit as st

from recommend import BookRecommender, UserRecommender
from generate import BooksDataFrameGenerator
from pathlib import Path


def main() -> None:

    st.title(":book: Book Recommendation Engine")
    st.header("Recommender")

    @st.cache_data
    def load_data():
        biblio = Path().cwd().parent.joinpath("data", "processed", "clean_df")
        biblio = pd.read_parquet(biblio)
        return biblio

    book_libray = load_data()

    if st.sidebar.checkbox("Show Dataframe", False):
        st.dataframe(book_libray)

    with st.sidebar:

        users, books = st.tabs(["Users", "Books"])

        with users:
            user_id = st.selectbox("Select User", book_libray["userId"].unique())
            num_recommendations = st.slider(
                "Number of Books to Recommend", min_value=1, max_value=10, value=5
            )

        with books:
            book_id = st.selectbox("Select Book", book_libray["bookId"].unique())
            num_recommendations = st.slider(
                "Recommend this book to how many users?", min_value=1, max_value=10, value=5
            )




if __name__ == "__main__":
    main()
