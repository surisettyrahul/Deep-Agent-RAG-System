from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from filereader import read_multiple_pdfs
from dotenv import load_dotenv

load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

DOC_PATHS = [
    "pdfs/climate_change.pdf",
    "pdfs/global_warming.pdf"
]

docs = read_multiple_pdfs(DOC_PATHS)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_splits = text_splitter.split_documents(docs)

print(f"split documentation into {len(all_splits)} chunks")

vector_store.add_documents(all_splits)

print(f"indexed {len(all_splits)} chunks")