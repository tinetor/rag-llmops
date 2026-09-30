"""Chunking: uses LangChain's text splitters to build heading-attached chunks,
merging small sections into their neighbor and splitting large ones."""

from pathlib import Path
import yaml

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

def chunking(raw_text: str, max_chars: int = 1500) -> list:
    """
    Generates text nodes based on character length limits without dropping any chunks,
    ensuring 100% of the raw text is preserved.
    """
    # Configure SentenceSplitter using len to measure chunk size in characters
    splitter = SentenceSplitter(
        chunk_size=max_chars,
        chunk_overlap=int(max_chars * 0.1),  # 10% character overlap
        tokenizer=len
    )
    
    # Wrap the raw text string into a LlamaIndex Document object
    documents = [Document(text=raw_text)]
    
    # Generate the list of TextNode objects explicitly for easy inspection/debugging
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Return nodes ready for the next pipeline step
    return nodes


if __name__ == "__main__":
    config = load_config()
    raw_text = Path("data/sample_corpus.txt").read_text()
    chunks = chunk_by_heading(
        raw_text,
        min_chars=config["chunking"]["min_chars"],
        max_chars=config["chunking"]["max_chars"],
    )
    print(f"{len(chunks)} chunks generated from data/sample_corpus.txt")
    for c in chunks:
        print(f"[{c['id']}] {c['heading']} ({len(c['text'])} chars)")
