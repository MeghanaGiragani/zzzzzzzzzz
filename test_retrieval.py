# =========================================================
# IMPORTS
# =========================================================

# Import pytest for writing and running test cases
import pytest
import sys
import os
# Import numpy (used for vector-related operations if needed)
import numpy as np

# Add parent directory (project root) to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import functions and class from retrieval_pipeline
from retrieval_pipeline import (
    retrieve_documents,   # Function to retrieve relevant documents
    generate_answer,      # Function to generate final answer
    cosine_similarity,    # Function to calculate similarity
    LocalEmbeddings       # Class to convert text into vectors
)


# =========================================================
# FAKE VECTOR STORE
# =========================================================

class FakeVectorStore:
    """
    This class simulates a vector database.
    Instead of using a real database, we use this
    to store documents for testing.
    """

    def __init__(self, docs):
        # Store the list of documents
        self.documents = docs


# =========================================================
# VALID TEST CASES
# =========================================================

def test_cosine_similarity_valid():
    """
    Test cosine similarity with valid vectors.
    Expected output = 0.5
    """

    test_name = "test_cosine_similarity_valid"   # Name of the test
    print(f"\n[{test_name}]")                   # Print test name

    expected_output = 0.5                       # Define expected result

    # Call the function with sample vectors
    actual_output = cosine_similarity([1, 0, 1], [1, 1, 0])

    # Print expected and actual values
    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {actual_output}")

    # Compare expected and actual using pytest.approx (for float comparison)
    assert actual_output == pytest.approx(expected_output), \
        f"[{test_name}] Failed → Expected {expected_output}, Got {actual_output}"


def test_retrieve_documents_valid():
    """
    Test retrieval of documents with valid input.
    Should return top 2 documents.
    """

    test_name = "test_retrieve_documents_valid"
    print(f"\n[{test_name}]")

    # Sample documents
    docs = [
        "RAG means Retrieval Augmented Generation",
        "Vector database stores embeddings"
    ]

    # Create fake vectorstore with documents
    store = FakeVectorStore(docs)

    expected_output = "2 documents"

    # Call retrieve function
    results = retrieve_documents("RAG", store, None, 2)

    # Print expected vs actual
    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {len(results)} documents")

    # Check length of results
    assert len(results) == 2, \
        f"[{test_name}] Failed → Expected 2, Got {len(results)}"


def test_generate_answer_valid():
    """
    Test generating answer using valid results.
    Output should be a non-empty string.
    """

    test_name = "test_generate_answer_valid"
    print(f"\n[{test_name}]")

    # Sample retrieval results (score, document)
    results = [
        (0.9, "RAG is Retrieval Augmented Generation."),
        (0.8, "Used for better answers")
    ]

    expected_output = "non-empty string"

    # Call function
    answer = generate_answer(results)

    # Print expected vs actual
    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {type(answer).__name__}")

    # Check if output is string
    assert isinstance(answer, str), \
        f"[{test_name}] Failed → Output is not string"


def test_embeddings_valid():
    """
    Test embedding generation from text.
    Should return two vectors.
    """

    test_name = "test_embeddings_valid"
    print(f"\n[{test_name}]")

    # Create embeddings object
    embeddings = LocalEmbeddings()

    # Convert text to vectors
    vectors = embeddings.embed_documents(["doc1", "doc2"])

    expected_output = "2 vectors"

    # Print expected vs actual
    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {len(vectors)} vectors")

    # Check number of vectors
    assert len(vectors) == 2, \
        f"[{test_name}] Failed → Expected 2, Got {len(vectors)}"


# =========================================================
# INVALID TEST CASES
# =========================================================

def test_invalid_query_type():
    """
    Test invalid query type.
    Should raise TypeError.
    """

    test_name = "test_invalid_query_type"
    print(f"\n[{test_name}]")

    # Create sample store
    store = FakeVectorStore(["sample"])

    embeddings = LocalEmbeddings()

    expected_output = "TypeError"

    # Expect TypeError when query is not string
    with pytest.raises(TypeError) as error:
        retrieve_documents(123, store, embeddings)

    # Capture actual error type
    actual_output = type(error.value).__name__

    # Print expected vs actual
    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {actual_output}")

    # Validate error type
    assert actual_output == expected_output


def test_empty_query():
    """
    Test empty query.
    Should raise ValueError.
    """

    test_name = "test_empty_query"
    print(f"\n[{test_name}]")

    store = FakeVectorStore(["sample"])
    embeddings = LocalEmbeddings()

    expected_output = "ValueError"

    # Expect ValueError for empty query
    with pytest.raises(ValueError) as error:
        retrieve_documents("", store, embeddings)

    actual_output = type(error.value).__name__

    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {actual_output}")

    assert actual_output == expected_output


def test_invalid_k_type():
    """
    Test invalid type for k.
    Should raise TypeError.
    """

    test_name = "test_invalid_k_type"
    print(f"\n[{test_name}]")

    store = FakeVectorStore(["sample"])
    embeddings = LocalEmbeddings()

    expected_output = "TypeError"

    # Pass string instead of integer
    with pytest.raises(TypeError) as error:
        retrieve_documents("query", store, embeddings, "2")

    actual_output = type(error.value).__name__

    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {actual_output}")

    assert actual_output == expected_output


def test_generate_answer_invalid_type():
    """
    Test invalid input type for generate_answer.
    Should raise TypeError.
    """

    test_name = "test_generate_answer_invalid_type"
    print(f"\n[{test_name}]")

    expected_output = "TypeError"

    # Pass string instead of list
    with pytest.raises(TypeError) as error:
        generate_answer("invalid")

    actual_output = type(error.value).__name__

    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {actual_output}")

    assert actual_output == expected_output


def test_cosine_similarity_invalid_input():
    """
    Test invalid input to cosine similarity.
    Should raise TypeError.
    """

    test_name = "test_cosine_similarity_invalid_input"
    print(f"\n[{test_name}]")

    expected_output = "TypeError"

    # Pass invalid datatype
    with pytest.raises(TypeError) as error:
        cosine_similarity("invalid", [1, 2, 3])

    actual_output = type(error.value).__name__

    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {actual_output}")

    assert actual_output == expected_output
