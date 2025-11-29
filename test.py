from src.database import initialize_database

# This triggers the model download (might take a minute the first time)
vector_store = initialize_database()
print("Database and Embeddings initialized successfully!")