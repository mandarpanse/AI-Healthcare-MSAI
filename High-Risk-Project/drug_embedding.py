# -----------------------------
# Section 1: Pretrained Drug Embeddings Extraction
# -----------------------------
# Download DRKG dataset from: https://github.com/gnn4dr/DRKG
# Direct link to embeddings: https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz

import numpy as np
import pandas as pd

# Load entity embeddings (from DRKG)
entity_embeddings = np.load('drkg/embed/DRKG_TransE_l2_entity.npy')

# Load entity names
with open('drkg/embed/entities.tsv', 'r') as f:
    entity_names = [line.strip() for line in f]

# Create DataFrame for embeddings
df_embeddings = pd.DataFrame(entity_embeddings)
df_embeddings.insert(0, 'entity_name', entity_names)

# Filter only drug entities (e.g., starting with 'Compound::')
drug_embeddings_df = df_embeddings[df_embeddings['entity_name'].str.startswith('Compound::')]

# Save drug embeddings to CSV
drug_embeddings_df.to_csv('drug_embeddings.csv', index=False)