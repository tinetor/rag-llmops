"""Chunking: groups raw text into heading-attached chunks, merging small
sections into their neighbor and splitting large ones."""

from pathlib import Path
import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def chunk_by_heading(raw_text, min_tokens=40, max_tokens=250):
    sections = []
    current_heading, current_text = None, []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("## "):
            if current_heading is not None:
                sections.append({"heading": current_heading, "text": " ".join(current_text).strip()})
            current_heading, current_text = line[3:].strip(), []
        elif line:
            current_text.append(line)
    if current_heading is not None:
        sections.append({"heading": current_heading, "text": " ".join(current_text).strip()})

    # merge small sections into the next one
    merged, buffer = [], None
    for sec in sections:
        if buffer is not None:
            buffer["text"] += " " + sec["text"]
            buffer["heading"] += " / " + sec["heading"]
            if len(buffer["text"].split()) >= min_tokens:
                merged.append(buffer)
                buffer = None
            continue
        if len(sec["text"].split()) < min_tokens:
            buffer = sec
        else:
            merged.append(sec)
    if buffer is not None:
        merged.append(buffer)

    # split large sections
    final_chunks = []
    for sec in merged:
        words = sec["text"].split()
        if len(words) <= max_tokens:
            final_chunks.append(sec)
        else:
            for i in range(0, len(words), max_tokens):
                final_chunks.append({"heading": sec["heading"], "text": " ".join(words[i:i + max_tokens])})
    for i, c in enumerate(final_chunks):
        c["id"] = i
    return final_chunks


if __name__ == "__main__":
    config = load_config()
    raw_text = Path("data/sample_corpus.txt").read_text()
    chunks = chunk_by_heading(
        raw_text,
        min_tokens=config["chunking"]["min_tokens"],
        max_tokens=config["chunking"]["max_tokens"],
    )
    print(f"{len(chunks)} chunks generated from data/sample_corpus.txt")
    for c in chunks:
        print(f"[{c['id']}] {c['heading']} ({len(c['text'].split())} words)")
