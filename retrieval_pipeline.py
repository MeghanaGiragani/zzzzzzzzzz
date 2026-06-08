# =========================================================

# Import required libraries
import pickle                     # Used to load saved vector database
import numpy as np               # Used for numerical operations
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text → vectors


# =========================================================
#  FUNCTION: LOAD VECTORSTORE
# =========================================================
def load_vectorstore():
    """
    This function loads the saved document database (pickle file)

    Input:
        None

    Output:
        Vectorstore object containing:
        - documents
        - vectors
    """

    # Open saved file in read-binary mode
    with open("db/store.pkl", "rb") as f:
        return pickle.load(f)


# =========================================================
# FUNCTION: COSINE SIMILARITY
# =========================================================
def cosine_similarity(v1, v2):
    """
    This function calculates similarity between two vectors

    Formula:
        cos(theta) = (v1 · v2) / (||v1|| * ||v2||)

    Input:
        v1 → First vector (list or numpy array)
        v2 → Second vector (list or numpy array)

    Output:
        Similarity score (float between 0 and 1)
    """

    # Check if inputs are valid types
    if not isinstance(v1, (list, np.ndarray)):
        raise TypeError("vec1 must be list or array")

    if not isinstance(v2, (list, np.ndarray)):
        raise TypeError("vec2 must be list or array")

    # Convert inputs to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Check empty vectors
    if len(v1) == 0 or len(v2) == 0:
        raise ValueError("Empty vector")

    # Check if vectors have same size
    if len(v1) != len(v2):
        raise ValueError("Unequal length")

    # Check if vector magnitude is zero
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        raise ValueError("Zero vector")

    # Apply cosine similarity formula
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # Return result as float
    return float(similarity)


# =========================================================
# CLASS: LOCAL EMBEDDINGS
# =========================================================
class LocalEmbeddings:
    """
    This class converts text into vectors using TF-IDF
    """

    def __init__(self):
        # Initialize TF-IDF model
        self.vectorizer = TfidfVectorizer()

        # Track whether model is trained
        self.fitted = False

    def embed_documents(self, docs):
        """
        Convert list of documents into vectors

        Input:
            docs → list of text

        Output:
            List of vectors
        """

        # Check input type
        if not isinstance(docs, list):
            raise TypeError("docs must be list")

        # Fit model and convert documents into vectors
        vectors = self.vectorizer.fit_transform(docs)

        # Mark model as fitted
        self.fitted = True

        # Return as list format
        return vectors.toarray().tolist()

    def embed_query(self, text):
        """
        Convert query into vector (after fitting)

        Input:
            text → string

        Output:
            vector → list
        """

        #  Ensure model is already trained
        if not self.fitted:
            raise ValueError("Model not fitted")

        # Convert query to vector
        vector = self.vectorizer.transform([text])

        return vector.toarray()[0].tolist()


# =========================================================
# FUNCTION: RETRIEVE DOCUMENTS
# =========================================================
def retrieve_documents(query, vectorstore, embeddings=None, k=2):
    """
    Retrieves top-k relevant documents for a given query

    Input:
        query → user question (string)
        vectorstore → contains documents
        embeddings → not used here (for compatibility)
        k → number of results

    Output:
        List of tuples → [(score, document), ...]
    """

    # =====================================================
    # INPUT VALIDATIONS
    # =====================================================

    # Query must be string
    if not isinstance(query, str):
        raise TypeError("Query must be string")

    # Query should not be empty
    if query.strip() == "":
        raise ValueError("Empty query")

    # Vectorstore should not be None
    if vectorstore is None:
        raise ValueError("Vectorstore None")

    # k must be integer
    if not isinstance(k, int):
        raise TypeError("k must be int")

    # k must be positive
    if k <= 0:
        raise ValueError("Invalid k")

    # Vectorstore must contain documents
    if not hasattr(vectorstore, "documents"):
        raise TypeError("Invalid vectorstore")

    documents = vectorstore.documents

    # Documents must be list
    if not isinstance(documents, list):
        raise TypeError("documents must be list")

    # Documents must not be empty
    if len(documents) == 0:
        raise ValueError("No documents")

    # All documents must be string
    if not all(isinstance(d, str) for d in documents):
        raise TypeError("documents must be strings")

    # =====================================================
    #  VECTOR CONVERSION
    # =====================================================

    # Combine documents + query
    all_text = documents + [query]

    # Convert into vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_text).toarray()

    # Last vector is query
    query_vec = vectors[-1]

    # Remaining are document vectors
    doc_vecs = vectors[:-1]

    # =====================================================
    # SIMILARITY COMPUTATION
    # =====================================================

    results = []

    for i, vec in enumerate(doc_vecs):
        score = cosine_similarity(query_vec, vec)
        results.append((score, documents[i]))

    # =====================================================
    # SORT RESULTS (HIGH → LOW)
    # =====================================================

    results.sort(reverse=True)

    # Return top-k results
    return results[:k]


# =========================================================
# FUNCTION: GENERATE ANSWER
# =========================================================
def generate_answer(results):
    """
    Combines retrieved documents into a final answer

    Input:
        results → [(score, document), ...]

    Output:
        Final answer (string)
    """

    # Validate input type
    if not isinstance(results, list):
        raise TypeError("results must be list")

    # Validate non-empty
    if len(results) == 0:
        raise ValueError("Empty results")

    answer = []

    for item in results:

        # Each result must be a tuple of size 2
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Invalid format")

        score, doc = item

        # Score must be numeric
        if not isinstance(score, (int, float)):
            raise TypeError("Score must be number")

        # Document must be string
        if not isinstance(doc, str):
            raise TypeError("Doc must be string")

        # Ignore empty documents
        if doc.strip() != "":
            answer.append(doc.strip())

    # Ensure final answer is not empty
    if len(answer) == 0:
        raise ValueError("Empty answer")

    # Join all documents into final output
    return "\n".join(answer)
