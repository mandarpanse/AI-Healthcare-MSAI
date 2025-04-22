
import streamlit as st
from datasets import load_dataset
import openai

from textstat import flesch_kincaid_grade, smog_index
import pandas as pd

# Load the Med-EASi dataset
@st.cache_data
def load_data():
    dataset = load_dataset("cbasu/Med-EASi", split="train")
    print(dataset)
    return dataset

# Prompt templates
def get_summarization_prompt(text):
    return f"Summarize the following medical note:\n\n{text}"

def get_simplification_prompt(text, audience):
    return f"Simplify the following medical note for a {audience}:\n\n{text}"

# Chain of Thought example
def get_cot_prompt(text):
    return (
        "Let's break down the medical note step by step and then summarize:\n"
        "1. Read the note carefully.\n"
        "2. Identify the key medical terms and procedures.\n"
        "3. Translate those into layman's terms.\n"
        "4. Create a short summary.\n\n"
        f"Note:\n{text}\n\nNow, the simplified summary:"
    )

# OpenAI API Call
def ask_openai(prompt):
    openai.api_key = "Update it appropriate value"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message["content"]



# Streamlit UI
def main():
    with st.container():
        st.title("LLM for Medical Text Simplification and Summarization")
        dataset = load_data()
        st.sidebar.title("Options")
        sample = st.sidebar.slider("Choose a sample by sliding a bar", 0, len(dataset)-1, 0)
        audience = st.sidebar.selectbox("Target Audience", ["child", "elderly", "non-native speaker", "general patient"])

        original_text = dataset[sample]["Expert"]
        st.subheader("Original Medical Note")
        st.write(original_text)

        col1, col2, col3, col4, col5 = st.columns([16,16,16,16,16])

        if st.button("Run"):

            st.write("")
            st.write("")
            st.write("")
            st.write("")
        
            # with st.container():
            with col1:
                summary_prompt = get_summarization_prompt(original_text)
                summary = ask_openai(summary_prompt)
                st.subheader("Summarized Text",  divider=True)
                st.write(summary)

            with col2:
                simplify_prompt = get_simplification_prompt(original_text, audience)
                simplified = ask_openai(simplify_prompt)
                st.subheader(f"Simplified for {audience}",  divider=True)
                st.write(simplified)

            with col3:
                cot_prompt = get_cot_prompt(original_text)
                cot_output = ask_openai(cot_prompt)
                st.subheader("Chain-of-Thought Output",  divider=True)
                st.write(cot_output)

            with col4:
                st.subheader("Original Text Readability",  divider=True)
                st.write(f"Flesch-Kincaid Grade : {flesch_kincaid_grade(original_text)} \n\n\n SMOG Index: {smog_index(original_text)}")

            with col5:
                st.subheader("Simplified Text Readability",  divider=True)
                st.write(f"Flesch-Kincaid Grade : {flesch_kincaid_grade(simplified)} \n\n\n SMOG Index: {smog_index(simplified)}")


if __name__ == "__main__":
    main()
