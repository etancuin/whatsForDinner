import kagglehub
import pandas as pd
import numpy as np
import os
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import torch
import faiss
from tqdm import tqdm

# Download dataset or load csv
if os.path.exists('recipes_data.csv'):
    print("Loading CSV...")
    df = pd.read_csv('recipes_data.csv')
else:
    print("Downloading dataset...")
    path = kagglehub.dataset_download("wilmerarltstrmberg/recipe-dataset-over-2m")
    print("Path to dataset files:", path)

    # Full path to the original CSV file in the downloaded dataset
    csv_file = os.path.join(path, "recipes_data.csv")
    
    print("Reading as CSV...")
    df = pd.read_csv(csv_file)

# Create embeddings or load existing ones
if os.path.exists('recipe_embeddings.npy'):
    print("Loading embeddings...")
    recipe_embeddings = np.load('recipe_embeddings.npy')
else:
    print("Creating embeddings...")
    recipe_texts = df["NER"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # Encode directly as float32
    recipe_embeddings = embed_model.encode(
        recipe_texts.tolist(), 
        show_progress_bar=True, 
        convert_to_numpy=True
    ).astype(np.float32)

    # Save in the correct format
    np.save('recipe_embeddings.npy', recipe_embeddings)

if os.path.exists('recipes_dataset'):
    dataset = Dataset.load_from_disk('recipes_dataset')
else:
    df = df.drop(columns=['link', 'site', 'source'])
    embeddings_list = []
    for embedding in recipe_embeddings:
        embeddings_list.append(embedding.tolist())  # Convert embeddings to list
    
    # Add embeddings to the dataframe
    df['embeddings'] = embeddings_list

    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk('recipes_dataset') 

# Create FAISS index using Hugging Face's datasets
print("Building FAISS index...")
dataset.add_faiss_index(column="embeddings")
print("FAISS index created successfully")

def test_ner_embeddings(embed_model, dataset, df):
    # Example ingredient-focused queries
    ingredient_queries = [
        "recipes with chocolate and butter",
        "dishes using rice and vegetables",
        "desserts with vanilla and sugar",
        "meals with chicken and garlic",
        "snacks with nuts"
    ]
    
    print("\n===== TESTING NER INGREDIENT-BASED EMBEDDINGS =====\n")
    
    for query in ingredient_queries:
        # Encode the query
        query_embedding = embed_model.encode([query])[0].astype(np.float32)
        
        # Search using Hugging Face's FAISS integration
        scores, retrieved_examples = dataset.get_nearest_examples(
            "embeddings", 
            query_embedding, 
            k=5
        )
        
        print(f"\nQuery: '{query}'")
        for i, (score, title, ingredients, directions) in enumerate(zip(
            scores,
            retrieved_examples["title"],
            retrieved_examples["ingredients"],
            retrieved_examples["directions"]
        )):
            print(f"Match {i+1}: {title} (Similarity: {score:.4f})")
            print(f"Ingredients: {ingredients[:200]}...")  # Truncated for long ingredient lists
            print(f"Directions: {directions[:200]}...")  # Truncated
            print("-" * 50)
        print("\n")

# Create the model instance for querying
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Run the test
test_ner_embeddings(embed_model, dataset, df)