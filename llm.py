from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Define the data path
DATA_PATH = "data/"

# Function to load PDF files from a directory
def load_pdf_files(data):
    try:
        loader = DirectoryLoader(data,
                                 glob="*.pdf",  # Load PDF files
                                 loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        print(f"Error loading PDF files: {e}")
        return []

# Load the documents
documents = load_pdf_files(data=DATA_PATH)

# Function to create text chunks from the loaded documents
def create_chunks(extracted_data):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(extracted_data)
        print(f"Created {len(text_chunks)} text chunks.")
        return text_chunks
    except Exception as e:
        print(f"Error creating chunks: {e}")
        return []

# Create chunks from the loaded documents
chunks = create_chunks(extracted_data=documents)

# Ensure there are chunks to process
if chunks:
    # Function to get the embedding model
    def get_embedding_model():
        try:
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            print(f"Loaded embedding model: {embedding_model}")
            return embedding_model
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            return None

    # Get the embedding model
    embedding_model = get_embedding_model()

    # Ensure the embedding model is loaded successfully
    if embedding_model:
        # Create FAISS vector store from text chunks and embedding model
        DB_FAISS_PATH = "vectorstore/db_faiss"
        try:
            db = FAISS.from_documents(chunks, embedding_model)
            db.save_local(DB_FAISS_PATH)
            print(f"FAISS database saved successfully at {DB_FAISS_PATH}")
        except Exception as e:
            print(f"Error saving FAISS database: {e}")
else:
    print("No chunks to process. Please check the PDF files.")
