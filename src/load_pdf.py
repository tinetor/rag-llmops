from llama_index.core import SimpleDirectoryReader

def load_pdf(pdf):
    # load documents
    raw_text = SimpleDirectoryReader(input_files=[pdf]).load_data()
    return raw_text




if __name__ == "__main__":
    result = load_pdf("data/ai_act.pdf")
    print(len(result), "documentos")
    print(repr(result[0].text[:300]))