import streamlit as st
import asyncio
import sys
import os

if sys.platform == "darwin":  # macOS only
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception as e:
        print("Asyncio event loop policy setup error:", e)


import pandas as pd
from chain_of_thoughts import generate_cot_response
from similarity import compute_similarity
from drug_id_mapping import drug_name_to_id
# from rag_process import rag_process
with st.container():
    st.title("Personalized DDI Risk Assessment")

    
    # cot_response = generate_cot_response(patient_info)
    # print(cot_response)

    age = st.number_input("Patient Age", min_value=0, max_value=120, value=70)
    # drug1 = st.text_area("Enter first drug name", "",)
    drug1 = st.selectbox("Select first drug", list(drug_name_to_id.keys()))
    # drug2 = st.text_area("Enter Second drug name", "")
    drug2 = st.selectbox("Select Second drug", list(drug_name_to_id.keys()))

    lab_results = st.text_area("Lab Results", "Elevated creatinine levels")


    if st.button("Assess Risk"):
        if (drug1 != drug2) :
            patient_profile = f"- Age: {age}\n- Medications: {drug1} and {drug2} \n- Lab Results: {lab_results}"
            embeddings_df = pd.read_csv('drug_embeddings.csv')

            # Example Similarity Check
            similarity = compute_similarity(drug1, drug2, embeddings_df)
            print(f"Similarity between {drug1} and {drug2}: {similarity}")

            # col1, filler, col2 = st.columns([2, 1, 6])
            # with col1:
            with st.container(height=150):
                st.write(f"### Similarity between {drug1} and {drug2}:")
                st.write(similarity)

            # rag_response = rag_process(drug1, drug2, lab_results)
            # print(f"{rag_response}")
            # st.write(f"### Reaction of two drugs:")
            # st.write(rag_response)

            # with col2:
            risk_container=st.container(height=300)
            with risk_container:
                response = generate_cot_response(patient_profile)

                risk_container.write("### Risk Assessment:")
                st.write(response)
