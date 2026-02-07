from google import genai
from dotenv import load_dotenv
import os
from llama_index.readers.file import PDFReader # this is for reading PDF files
from llama_index.core.node_parser import SentenceSplitter # this is for splitting text into sentences

load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMENI_API"))
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM=3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents = texts
    )
    return [item.values for item in response.embeddings]