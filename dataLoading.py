import os
import sys
import io
import json
import hashlib
from pathlib import Path
from datetime import datetime

sys.stderr = io.StringIO()

os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; YourBot/1.0)"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
import glob
import warnings
import logging

# ----------------------------
# CLEAN TERMINAL OUTPUT
# ----------------------------
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("PyPDF2").setLevel(logging.ERROR)

# ----------------------------
# CONFIGURATION
# ----------------------------
PDF_FOLDER = "data/pdfs/"
URLS_FILE = "data/urls.txt"
VECTORSTORE_PATH = "SL_Laws_faiss"
METADATA_FILE = "vectorstore_metadata.json"


# ----------------------------
# METADATA MANAGEMENT
# ----------------------------
def get_files_hash(pdf_folder, urls_file):
    """Generate hash of all PDF files and URLs file for change detection"""
    hash_md5 = hashlib.md5()

    # Hash PDF files (name + modification time)
    pdf_files = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")))
    for pdf_path in pdf_files:
        # Include filename and modification time
        hash_md5.update(pdf_path.encode())
        hash_md5.update(str(os.path.getmtime(pdf_path)).encode())

    # Hash URLs file content
    if os.path.exists(urls_file):
        with open(urls_file, 'r') as f:
            hash_md5.update(f.read().encode())
        hash_md5.update(str(os.path.getmtime(urls_file)).encode())

    return hash_md5.hexdigest()


def load_metadata():
    """Load existing metadata if it exists"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return None


def save_metadata(files_hash, num_pdfs, num_urls, num_chunks):
    """Save metadata about the current vectorstore"""
    metadata = {
        "files_hash": files_hash,
        "num_pdfs": num_pdfs,
        "num_urls": num_urls,
        "num_chunks": num_chunks,
        "last_build": datetime.now().isoformat(),
        "vectorstore_path": VECTORSTORE_PATH
    }
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    return metadata


def should_rebuild_vectorstore():
    """Check if vectorstore needs to be rebuilt"""
    # Check if vectorstore exists
    if not os.path.exists(VECTORSTORE_PATH):
        print("📦 No existing vectorstore found. Building new one...")
        return True

    # Check if metadata exists
    metadata = load_metadata()
    if not metadata:
        print("📦 No metadata found. Rebuilding vectorstore...")
        return True

    # Check if files have changed
    current_hash = get_files_hash(PDF_FOLDER, URLS_FILE)
    if current_hash != metadata.get("files_hash"):
        print("📦 Files have changed. Rebuilding vectorstore...")
        return True

    print("✅ Vectorstore is up to date. Loading from cache...")
    return False


# ----------------------------
# DOCUMENT LOADING
# ----------------------------
def load_multiple_pdfs(file_paths):
    """Load multiple PDF files"""
    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs


def load_urls_from_file(file_path):
    """Load documents from URLs listed in a file"""
    if not os.path.exists(file_path):
        print(f"⚠️  URLs file not found: {file_path}")
        return []

    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    all_docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"⚠️  Failed to load {url}: {str(e)}")

    return all_docs


# ----------------------------
# CHUNKING
# ----------------------------
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)


# ----------------------------
# EMBEDDINGS
# ----------------------------
def create_embeddings():
    """Create HuggingFace embeddings"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-V2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


# ----------------------------
# VECTORSTORE CREATION
# ----------------------------
def create_faiss_vectorstore(chunks, embeddings):
    """Create FAISS vectorstore from chunks"""
    return FAISS.from_documents(chunks, embeddings)


def build_vectorstore():
    """Build vectorstore from scratch"""
    print("\n" + "=" * 80)
    print("🔨 BUILDING VECTORSTORE FROM SCRATCH")
    print("=" * 80)

    # Load PDFs
    pdf_files = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
    documents = load_multiple_pdfs(pdf_files)
    print(f"📄 Loaded {len(documents)} pages from {len(pdf_files)} PDFs")

    # Load URLs
    url_documents = load_urls_from_file(URLS_FILE)
    print(f"🌐 Loaded {len(url_documents)} documents from URLs")

    # Combine all documents
    all_documents = documents + url_documents
    print(f"📚 Total documents: {len(all_documents)}")

    # Chunk documents
    text_chunks = chunk_documents(all_documents)
    print(f"✂️  Created {len(text_chunks)} chunks")

    # Create embeddings
    print("🧮 Creating embeddings...")
    embeddings = create_embeddings()

    # Create vectorstore
    print("💾 Building FAISS vectorstore...")
    vectorstore = create_faiss_vectorstore(text_chunks, embeddings)

    # Save vectorstore
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"✅ Vectorstore saved to {VECTORSTORE_PATH}")

    # Save metadata
    files_hash = get_files_hash(PDF_FOLDER, URLS_FILE)
    metadata = save_metadata(files_hash, len(pdf_files), len(url_documents), len(text_chunks))
    print(f"📝 Metadata saved: {metadata['num_chunks']} chunks, built at {metadata['last_build']}")
    print("=" * 80 + "\n")

    return vectorstore, embeddings


def load_vectorstore():
    """Load existing vectorstore from disk"""
    print("\n" + "=" * 80)
    print("⚡ LOADING CACHED VECTORSTORE")
    print("=" * 80)

    embeddings = create_embeddings()
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    metadata = load_metadata()
    if metadata:
        print(f"📊 Loaded vectorstore with {metadata['num_chunks']} chunks")
        print(f"📅 Last built: {metadata['last_build']}")

    print("✅ Vectorstore loaded successfully!")
    print("=" * 80 + "\n")

    return vectorstore, embeddings


# ----------------------------
# MAIN INITIALIZATION
# ----------------------------
def initialize_vectorstore(force_rebuild=False):
    """Initialize vectorstore with caching"""
    if force_rebuild or should_rebuild_vectorstore():
        return build_vectorstore()
    else:
        return load_vectorstore()


# Initialize on import
vectorstore, embeddings = initialize_vectorstore()

# ----------------------------
# EXPORT
# ----------------------------
__all__ = ['vectorstore', 'embeddings', 'initialize_vectorstore']