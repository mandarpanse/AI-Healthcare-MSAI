from sklearn.metrics.pairwise import cosine_similarity
from drug_id_mapping import drug_name_to_id
import openai

def get_drug_embedding(drug_name, embeddings_df):
    row = embeddings_df[embeddings_df['entity_name'].str.contains(drug_name, case=False)]
    if not row.empty:
        return row.iloc[0, 1:].values.astype(float)
    else:
        return None

def compute_similarity(drug1, drug2, embeddings_df):
    mapped_drug1 = drug_name_to_id.get(drug1)
    mapped_drug2 = drug_name_to_id.get(drug2)

    emb1 = get_drug_embedding(str(mapped_drug1), embeddings_df)
    emb2 = get_drug_embedding(str(mapped_drug2), embeddings_df)

    print(f"{drug1}/{emb1}")
    print(f"{drug2}/{emb2}")
    if emb1 is not None and emb2 is not None:
        
        cos_similarity = cosine_similarity([emb1], [emb2])[0][0]
        similarity_description = describe_similarity(drug1, drug2, cos_similarity)
        return similarity_description
    else:
        return None
    
def describe_similarity(drug1, drug2, cos_similarity):
    openai.api_key = 'Update your openAI API Key'

    prompt = f"""

Cosine Similarity of {drug1} and {drug2} is: {cos_similarity}

Question:
Describe the similarity between two given drugs based on given cosine similarity also in terms of their composition in 3 to 4 sentences.

Answer:
Let's think step by step.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a clinical pharmacologist providing detailed reasoning for drug interactions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    print(f"response of COT - {response}")
    return response['choices'][0]['message']['content']

