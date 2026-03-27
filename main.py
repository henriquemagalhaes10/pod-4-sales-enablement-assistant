import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.chains.summarize import load_summarize_chain

# Load environment variables
load_dotenv()

# --- Unified Configurations ---
AZURE_ENDPOINT = "https://ai-bootcamp-openai-pod4.openai.azure.com/"
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
CHAT_DEPLOYMENT = "gpt-4o-mini"
# API Versions: Using your specified versions for each client
LANGCHAIN_API_VERSION = "2024-02-01"
DIRECT_CLIENT_API_VERSION = "2024-12-01-preview"

# ==========================================
# PART 1: Direct Azure OpenAI Client (Testing)
# ==========================================
print("--- Testing Direct Embedding Connection ---")
client = AzureOpenAI(
    api_version=DIRECT_CLIENT_API_VERSION,
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
)

test_response = client.embeddings.create(
    input=["first phrase", "second phrase"],
    model=EMBEDDING_DEPLOYMENT
)

for item in test_response.data:
    print(f"Direct Client Vector {item.index} | Length: {len(item.embedding)}")

# ==========================================
# PART 2: LangChain RAG Pipeline
# ==========================================

# 1. Initialize LangChain Azure Objects
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDING_DEPLOYMENT,
    openai_api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=LANGCHAIN_API_VERSION,
)

llm = AzureChatOpenAI(
    azure_deployment=CHAT_DEPLOYMENT,
    openai_api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=LANGCHAIN_API_VERSION,
    temperature=0
)

# 2. Process PDF
print("\n--- Loading and Chunking PDF ---")
loader = PyPDFLoader("document.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# 3. Upload to Qdrant
print(f"--- Uploading {len(chunks)} chunks to Qdrant ---")
qdrant = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="azure_pdf_summaries",
    force_recreate=True
)

# 4. Generate Summary
print("--- Generating Summary (Map-Reduce) ---")
summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = summary_chain.invoke(chunks)

print("\n--- Final Document Summary ---")
print(summary["output_text"])
