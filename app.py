import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Movie Revenue Prediction", page_icon="🎬")
st.write("""
Predict the expected movie revenue using a Machine Learning model trained on historical movie data.
""")
st.write("Predict the expected movie revenue using machine learning.")
st.sidebar.title("About")

st.sidebar.info("""
Movie Revenue Prediction App

Built using:
• Python
• Streamlit
• Machine Learning
• Scikit-learn
""")
model = joblib.load("movie_revenue.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

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

director = st.selectbox(
    "Director Name",
    encoders["director"].classes_
)

writer = st.selectbox(
    "Writer Name",
    encoders["writer"].classes_
)

star = st.selectbox(
    "Lead Actor/Actress",
    encoders["star"].classes_
)

country = st.selectbox(
    "Country",
    encoders["country"].classes_
)

company = st.selectbox(
    "Production Company",
    encoders["company"].classes_
)

budget = st.number_input(
    "Budget",
    min_value=0.0,
    value=1000000.0
)

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
        st.balloons()
        st.markdown("---")
        st.caption("Created by Padma,Priya,Ragha")
    except ValueError:
        st.error(
            "One or more values (Director, Writer, Star, Country, or Company) "
            "were not found in the training data. Please enter a value that exists in the dataset."
        )
