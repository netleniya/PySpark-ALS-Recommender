# Book Recommendation System using ALS in Spark

## Objective

This code provides an overview of a Book Recommendation System project built using Apache Spark's Alternating Least Squares (ALS) algorithm. The project aims to provide users with personalized book recommendations based on their past reading and rating behavior. The system is implemented as a web application using Gradio.



## Data Preparation
The dataset comprises 3 files: `Users`, `Books` and `Ratings`, and was acquired on the website GitHub at the following address: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset 


## Features
- **Data Processing:** The raw data is batch-processed using `Dask` and cleaned to remove missing values and books/ratings with fewer than 10 ratings. The resulting dataframe is exported as a parquet file to be fed into the Spark ALS algorithm.
- **Model Training:** The `Spark ALS` algorithm is used to train a model on the processed data (see the `eda.ipynb` notebook).
- **Model Evaluation:** The ALS model is evaluated using the RMSE metric (see the `sparkml.ipynb` notebook).
- **Model Serving:** The model is deployed as a web application using [Gradio](https://gradio.app/).

## Getting Started
1. Clone the repository: `git clone git@github.com:netleniya/spark_recommender.git`
2. Process and clean the data: `python eda.py`
3. Train the ALS model: `python sparkml.py`, and export the model as a binary
4. Generate predictions using the web app: `gradio main.py` or `python main.py`

## License

This project is licensed under the [MIT License](LICENSE).

## Original project in collaboration with
- Bomal, Laure Anna
- Chun, Steve
- Feng, Haoyue
- Fuenmayor Mejia, Alexander
- Mak, Jimmy