import pandas as pd
import gradio as gr

from recommend import BookRecommender, UserRecommender
from generate import GenerateBookList, GenerateUsersList
from pathlib import Path


def load_data() -> pd.DataFrame:
    biblio = Path(__file__).parents[1].joinpath("data", "processed", "clean_df")
    biblio = pd.read_parquet(biblio)
    return biblio


book_libray = load_data()


def get_book_recommendations(user_id, num_recs) -> pd.DataFrame:

    try:
        rec_obj = BookRecommender().recommend(num_recommendations=num_recs)
        gen_obj = GenerateBookList(book_libray, rec_obj).generate_dataframe(
            user_id=user_id
        )
        return gen_obj
    except ValueError as e:
        print(f"No recommendations for {user_id} or User not found")


def get_user_recommendations(book_id, num_recs) -> pd.DataFrame:

    try:
        rec_obj = UserRecommender().recommend(num_recommendations=num_recs)
        gen_obj = GenerateUsersList(book_libray, rec_obj).generate_dataframe(
            isbn=book_id
        )
        return gen_obj
    except ValueError as e:
        print(f"No recommendations for {book_id} or Book not found")


def main() -> None:

    book_rec = gr.Interface(
        fn=get_book_recommendations,
        inputs=[
            gr.Number(label="User ID"),
            gr.Number(label="Number of recommendations"),
        ],
        outputs=[
            gr.DataFrame(
                label="Recommended Books",
                col_count=(3, "fixed"),
            ),
        ],
        title="Book Recommender",
    )

    user_rec = gr.Interface(
        fn=get_user_recommendations,
        inputs=[
            gr.Number(label="Book ID"),
            gr.Number(label="Number of recommendations"),
        ],
        outputs=[
            gr.DataFrame(
                label="Recommended Users",
                col_count=(3, "fixed"),
            ),
        ],
        title="User Recommender",
    )

    interface = gr.TabbedInterface(
        [book_rec, user_rec], ["Book Recommender", "User Recommender"]
    )
    interface.launch()


if __name__ == "__main__":
    main()
