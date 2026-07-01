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

if st.button("Predict Revenue"):

    try:

        rating_encoded = encoders["rating"].transform([rating])[0]
        genre_encoded = encoders["genre"].transform([genre])[0]
        director_encoded = encoders["director"].transform([director])[0]
        writer_encoded = encoders["writer"].transform([writer])[0]
        star_encoded = encoders["star"].transform([star])[0]
        country_encoded = encoders["country"].transform([country])[0]
        company_encoded = encoders["company"].transform([company])[0]

        input_data = pd.DataFrame([[

            favorability,
            rating_encoded,
            genre_encoded,
            year,
            votes,
            director_encoded,
            writer_encoded,
            star_encoded,
            country_encoded,
            budget,
            company_encoded,
            runtime

        ]], columns=[

            "favorability",
            "rating",
            "genre",
            "year",
            "votes",
            "director",
            "writer",
            "star",
            "country",
            "budget",
            "company",
            "runtime"

        ])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        st.success(f"🎉 Predicted Movie Revenue: ${prediction[0]:,.2f}")

    except ValueError:
        st.error(
            "One or more values (Director, Writer, Star, Country, or Company) "
            "were not found in the training data. Please enter a value that exists in the dataset."
        )
