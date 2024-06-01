from recommend import BookRecommender, ReaderRecommender
from pathlib import Path
import streamlit as st


def main() -> None:
    st.title(":book: Book Recommendations")
    st.write("This is a demo of the recommendations app.")


if __name__ == "__main__":
    main()
