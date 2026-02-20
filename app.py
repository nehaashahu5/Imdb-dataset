import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


# -----------------------------
# Title
# -----------------------------
st.title("IMDB Sentiment Analysis")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

test_size = st.sidebar.slider(
    "Test Size",
    0.1, 0.5, 0.2, 0.05
)

max_features = st.sidebar.slider(
    "Max Features (TF-IDF)",
    1000, 10000, 5000, 500
)


# -----------------------------
# Upload dataset
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload IMDB Dataset CSV",
    type="csv"
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Expecting columns: review, sentiment
    # sentiment -> positive / negative
    df["sentiment"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    st.subheader("Dataset Preview")
    st.write(df.head())


    # -----------------------------
    # Train test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=test_size,
        random_state=42
    )


    # -----------------------------
    # TF-IDF
    # -----------------------------
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=max_features
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)


    # -----------------------------
    # Model
    # -----------------------------
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)


    # -----------------------------
    # Prediction & accuracy
    # -----------------------------
    y_pred = nb.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)

    st.subheader("Model Accuracy")
    st.write(f"Accuracy : {acc:.4f}")


    # -----------------------------
    # Confusion matrix
    # -----------------------------
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="mako",
        ax=ax
    )

    ax.set_title("Sentiment Analysis Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)


    # -----------------------------
    # Test a review
    # -----------------------------
    st.subheader("Test a Review")

    user_review = st.text_input("Enter a review")

    if user_review:

        vec = tfidf.transform([user_review])
        pred = nb.predict(vec)[0]

        if pred == 1:
            st.success("Sentiment : Positive")
        else:
            st.error("Sentiment : Negative")
