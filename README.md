Repository for Natural Language Processing project.

Collaborators: Maria Amelia Morsellino, Alessia Mercadante

## Dataset
Source: Food.com (https://doi.org/10.34740/kaggle/dsv/783630)
Files expected in project root:
- `RAW_recipes.csv`
- `RAW_interactions.csv`

The code supports a **test mode** to work on a subset for practicality:
```
TEST_MODE = True
SAMPLE_SIZE = 100000
```

## Recipe Retrieval: TF-IDF vs Neural Embeddings
This repository contains a recipe retrieval project that builds and compares two search engines over the Food.com dataset.

### TF-IDF (Lexical Search)
The TF-IDF engine represents each recipe as a single text document obtained by concatenating **name, tags, description, ingredients, and steps**.  
Before vectorization, the text undergoes **strong preprocessing** (NLTK): lowercasing, removal of non-alphabetic characters, tokenization, **english + recipe-domain stopwords** and **WordNet lemmatization**.
Documents are then indexed using `TfidfVectorizer` with **unigrams + bigrams**. At query time, the query is processed in the same way and results are ranked using **cosine similarity** against the TF-IDF matrix.

### Semantic Search (Neural Embeddings)
The semantic engine encodes recipes and queries into **dense vector embeddings** using `SentenceTransformer` (`all-MiniLM-L6-v2`).  
Each recipe is formatted into a semantic-friendly string to preserve key contextual signals. Embeddings are computed in batches and **L2-normalized**, so cosine similarity is efficiently computed via **dot product**.  
This approach is typically more robust on **conceptual or descriptive queries**, capturing semantic similarity beyond exact keyword matches.

### Evaluation
Evaluation is performed in two steps:
1. **Top-K overlap analysis**: for a curated set of queries (keyword, semantic, dietary/cuisine), both engines retrieve Top-K results. The project measures how many recipe IDs overlap between the two ranked lists, both overall and by query category.
2. **Manual relevance judgments + Precision@5**: a CSV template is generated containing the Top-5 results for each query and engine. After manually labeling each item as relevant (0-1), the script computes **Precision@5** per query/engine, aggregates results by category, identifies per-query winners, and exports the final metrics to `evaluation_results.csv`.

### Dependencies
```
pip install pandas numpy scikit-learn torch sentence-transformers nltk tqdm
```
