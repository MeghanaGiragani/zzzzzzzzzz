# =========================================================

# Import required libraries
import os                      # Used to work with file system (folders/files)
import pickle                  # Used to save data in binary format
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text into vectors


# =========================================================
# CLASS: SIMPLE VECTOR STORE
# =========================================================
class SimpleVectorStore:
    """
    This class acts as a simple database.

    It stores:
    - documents → original text data
    - vectors → numerical representation of text
    """

    def __init__(self):
        # Initialize empty lists
        self.documents = []
        self.vectors = []

    def add(self, docs, vectors):
        """
        Store documents and their corresponding vectors

        Input:
            docs → list of text documents
            vectors → list of numerical vectors
        """
        self.documents = docs
        self.vectors = vectors

    def save(self, path):
        """
        Save the data into a file using pickle

        Input:
            path → location to save the file

        Output:
            Creates a file (e.g., db/store.pkl)
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)


# =========================================================
# FUNCTION: RUN INGESTION
# =========================================================
def run_ingestion():
    """
    This function performs full ingestion process:

    Step 1 → Load documents from 'docs' folder
    Step 2 → Convert documents into vectors using TF-IDF
    Step 3 → Store and save them in vector database

    Final Output:
        Saved file: db/store.pkl
    """

    print("Starting ingestion...")

    # =====================================================
    # STEP 1: LOAD DOCUMENTS
    # =====================================================

    docs = []  # Empty list to store document contents

    # Loop through all files in 'docs' folder
    for file in os.listdir("docs"):

        # Only read .txt files
        if file.endswith(".txt"):

            # Open file and read content
            with open(os.path.join("docs", file), "r", encoding="utf-8") as f:
                docs.append(f.read())

    # Optional validation (good practice)
    if len(docs) == 0:
        raise ValueError("No documents found in docs folder")

    print(f"Loaded {len(docs)} documents")


    # =====================================================
    #STEP 2: CONVERT TEXT → VECTORS
    # =====================================================

    # Initialize TF-IDF model
    vectorizer = TfidfVectorizer()

    # Convert documents into numerical vectors
    vectors = vectorizer.fit_transform(docs).toarray()

    print("Documents converted into vectors")


    # =====================================================
    # STEP 3: STORE DATA
    # =====================================================

    # Create vector store object
    store = SimpleVectorStore()

    # Add documents + vectors into store
    store.add(docs, vectors)

    print("Data stored in vectorstore")


    # =====================================================
    # STEP 4: SAVE TO FILE
    # =====================================================

    # Create db folder if not exists
    os.makedirs("db", exist_ok=True)

    # Save file
    store.save("db/store.pkl")

    print("Ingestion complete and saved to db/store.pkl")
