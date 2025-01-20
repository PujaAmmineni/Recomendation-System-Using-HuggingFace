# Recomendation-System-Using-HuggingFace

## Overview

This project implements a content-based recommendation system using sentence-transformers. It generates semantic embeddings for anime descriptions and performs recommendations based on cosine similarity.

## Features

- **Content-Based Recommendation**: Recommends anime based on textual similarity of their descriptions.
- **Sentence Embeddings**: Uses pre-trained sentence-transformer models for high-quality text embeddings.
- **Query Matching**: Provides relevant results based on user-inputted queries.

## Dataset

The dataset contains information about various anime, including:

- **name**: The title of the anime.
- **genre**: The genres associated with the anime.
- **type**: The type (e.g., TV, Movie, ONA).
- **episodes**: The number of episodes.
- **rating**: User ratings.
- **members**: Number of members who have rated or interacted with the anime.

The dataset is processed to create a new column, `description`, which combines the `name`, `genre`, `type`, and `episodes` fields.

## Technologies Used

### Python Libraries:

- `pandas` for data manipulation.
- `sentence-transformers` for generating sentence embeddings.
- `scikit-learn` for cosine similarity calculation.
- `torch` for efficient tensor computations.

## Installation

1. Clone the repository and navigate to the project directory.
2. Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Example dependencies:

- `pandas`
- `sentence-transformers`
- `scikit-learn`
- `torch`

## Usage

### Data Preparation:

1. Load the anime dataset (`anime.csv`).
2. Preprocess the data to remove missing values and generate descriptions.

```python
import pandas as pd

df = pd.read_csv('anime.csv')
df = df.dropna()
df['description'] = df['name'] + ' ' + df['genre'] + ' ' + df['type'] + ' episodes: ' + df['episodes']
```

### Generate Embeddings:

Use the `sentence-transformers` library to encode anime descriptions into embeddings.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
anime_embeddings = model.encode(df['description'].tolist())
```

### Recommendation Query:

Use the `get_recommendations` function to retrieve the top recommendations for a given query.

```python
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(query, embeddings, df, top_n=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices]

query = "horror movie"
recommendations = get_recommendations(query, anime_embeddings, df)
print(recommendations[['name', 'genre', 'description']])
```

### Example Output

For the query `"horror movie"`, the system returns anime recommendations ranked by similarity:

| Name                   | Genre           | Description                              |
|------------------------|-----------------|------------------------------------------|
| Zonmi-chan: Halloween Movie | Comedy, Horror | Zonmi-chan: Halloween Movie Comedy, Horror ... |
| Darkside Blues         | Horror, Mystery, Sci-Fi | Darkside Blues Horror, Mystery, Sci-Fi Movie episodes... |
| Seoul-yeok             | Horror, Thriller | Seoul-yeok Horror, Thriller Movie episodes: 1 |

## Improvements

- **Performance**: Use GPU acceleration and pre-compute embeddings for faster runtime.
- **Scalability**: Integrate approximate nearest neighbor (ANN) libraries like FAISS for large datasets.
- **Diversity**: Implement filtering to ensure diversity in recommendations.
- **User Interface**: Build a web or mobile interface for end-users.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for pre-trained models.
- [SentenceTransformers](https://www.sbert.net/) library for efficient sentence embeddings.
