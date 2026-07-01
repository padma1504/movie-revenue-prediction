import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("movie_revenue.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Movie Revenue Prediction", page_icon="🎬")

st.title("🎬 Movie Revenue Prediction")
st.write("Enter the movie details below to predict the revenue.")

favorability = st.number_input(
    "Favorability Score",
    min_value=0.0,
    value=0.5
)

rating = st.selectbox(
    "Movie Rating",
    list(encoders["rating"].classes_)
)

genre = st.selectbox(
    "Genre",
    list(encoders["genre"].classes_)
)

year = st.number_input(
    "Release Year",
    min_value=1900,
    max_value=2100,
    value=2024
)

votes = st.number_input(
    "Votes",
    min_value=0,
    value=1000
)

director = st.text_input("Director Name")

writer = st.text_input("Writer Name")

star = st.text_input("Lead Actor/Actress")

country = st.text_input("Country")

budget = st.number_input(
    "Budget",
    min_value=0.0,
    value=1000000.0
)

company = st.text_input("Production Company")

runtime = st.number_input(
    "Runtime (minutes)",
    min_value=1,
    value=120
)
