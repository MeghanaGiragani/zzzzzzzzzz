# =========================================================
# IMPORTS
# =========================================================

# pytest is used for writing and running test cases
import pytest
import sys
# os is used for file and folder operations
import os

# shutil is used to delete folders/files after testing
import shutil
# Add parent directory (project root) to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import ingestion functions and class from ingestion_pipeline
from ingestion_pipeline import (
    run_ingestion,        # main ingestion function
    SimpleVectorStore     # class to store documents and vectors
)


# =========================================================
# HELPER FUNCTIONS (SETUP & CLEANUP)
# =========================================================

def create_test_docs():
    """
    This function creates a temporary 'docs' folder
    with sample text files for testing ingestion.
    """

    # Create docs folder if it does not exist
    os.makedirs("docs", exist_ok=True)

    # Create first test file
    with open("docs/file1.txt", "w") as f:
        f.write("RAG stands for Retrieval Augmented Generation")

    # Create second test file
    with open("docs/file2.txt", "w") as f:
        f.write("TF-IDF converts text into vectors")


def cleanup():
    """
    This function deletes test folders after execution
    to keep environment clean.
    """

    # Remove docs folder if exists
    if os.path.exists("docs"):
        shutil.rmtree("docs")

    # Remove db folder if exists
    if os.path.exists("db"):
        shutil.rmtree("db")


# =========================================================
# VALID TEST CASES
# =========================================================

def test_ingestion_valid():
    """
    Test complete ingestion pipeline with valid data.

    Expected:
    - store.pkl file should be created in db folder
    """

    test_name = "test_ingestion_valid"
    print(f"\n[{test_name}]")

    # Create sample documents
    create_test_docs()

    expected_output = "store.pkl created"

    # Run ingestion pipeline
    run_ingestion()

    # Check if file was created
    file_exists = os.path.exists("db/store.pkl")

    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {'store.pkl created' if file_exists else 'not created'}")

    # Assert file created successfully
    assert file_exists, f"[{test_name}] Failed → store.pkl not created"

    # Clean up test environment
    cleanup()


def test_vectorstore_storage():
    """
    Test whether SimpleVectorStore stores documents correctly.

    Expected:
    - documents list should contain 2 items
    """

    test_name = "test_vectorstore_storage"
    print(f"\n[{test_name}]")

    # Create store object
    store = SimpleVectorStore()

    # Sample input data
    docs = ["doc1", "doc2"]
    vectors = [[1, 2], [3, 4]]

    # Add data to store
    store.add(docs, vectors)

    expected_output = "2 documents"

    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {len(store.documents)} documents")

    # Check if documents stored correctly
    assert len(store.documents) == 2, \
        f"[{test_name}] Failed → Expected 2, Got {len(store.documents)}"


# =========================================================
# INVALID TEST CASES
# =========================================================

def test_missing_docs_folder():
    """
    Test ingestion when 'docs' folder does not exist.

    Expected:
    - Error should be raised
    """

    test_name = "test_missing_docs_folder"
    print(f"\n[{test_name}]")

    # Ensure docs folder is removed
    cleanup()

    expected_output = "FileNotFoundError or ValueError"

    # Run ingestion and expect an error
    with pytest.raises(Exception) as error:
        run_ingestion()

    actual_output = type(error.value).__name__

    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {actual_output}")

    # Check if error type is valid
    assert actual_output in ["FileNotFoundError", "ValueError"]


def test_empty_docs_folder():
    """
    Test ingestion when docs folder exists but is empty.

    Expected:
    - ValueError should be raised
    """

    test_name = "test_empty_docs_folder"
    print(f"\n[{test_name}]")

    # Create empty docs folder
    os.makedirs("docs", exist_ok=True)

    expected_output = "ValueError"

    # Run ingestion and expect ValueError
    with pytest.raises(ValueError) as error:
        run_ingestion()

    actual_output = type(error.value).__name__

    print(f"Expected Output: {expected_output}")
    print(f"Actual Output  : {actual_output}")

    # Validate error type
    assert actual_output == expected_output

    # Clean up
    cleanup()
