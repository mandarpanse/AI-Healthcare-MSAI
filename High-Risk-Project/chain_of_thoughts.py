# -----------------------------
# Section 3: Chain-of-Thought Prompting with OpenAI
# -----------------------------
import openai

openai.api_key = 'Update your openAI API Key'

def generate_cot_response(patient_info):
    prompt = f"""
Patient Profile:
{patient_info}

Question:
Assess the risk of drug-drug interactions and provide recommendations.

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