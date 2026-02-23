import streamlit as st
import pdfplumber

st.title("IMDB Dataset – Direct Output")

PDF_PATH = "/mnt/data/Imdb dataset.pdf"


@st.cache_data
def read_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


try:
    data = read_pdf(PDF_PATH)

    st.subheader("Direct Output from PDF (Synthetic view / preview)")
    st.text_area("Dataset Text", data[:5000], height=400)

    st.success("PDF directly app.py se read ho gayi.")

except Exception as e:
    st.error("PDF read nahi ho pa rahi.")
    st.write(e)