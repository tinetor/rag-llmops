

from pathlib import Path
from config import load_config
from load_pdf import load_pdf
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter





def chunking(raw_text: list, max_chars: int = 1500) -> list:
    """
    Generates text nodes based on character length limits without dropping any chunks,
    ensuring 100% of the raw text is preserved.
    """
    for doc in raw_text:
        doc.excluded_embed_metadata_keys = list(doc.metadata.keys())    
    # Configure SentenceSplitter using len to measure chunk size in characters
    splitter = SentenceSplitter(
        chunk_size=max_chars,
        chunk_overlap=int(max_chars * 0.1)  # 10% character overlap
        
    )
    
  
    
    # Generate the list of TextNode objects
    nodes = splitter.get_nodes_from_documents(raw_text)
    
    # Return all generated nodes directly
    return nodes


if __name__ == "__main__":
    config = load_config()
    
    raw_text = load_pdf(config["data"]["pdf_path"])
    chunks = chunking(raw_text)
    print(chunks[0].metadata)
    print(f"{len(chunks)} chunks generated from {config['data']['pdf_path']}")
    for i, c in enumerate(chunks):
        print(f"[{i}] page {c.metadata.get('page_label')} | {len(c.text)} chars | {c.text[:80]!r}")