from langchain_community.document_loaders import PyPDFLoader

def inspect_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    print(f"Total pages: {len(pages)}")
    
    with open("pdf_content.txt", "w") as f:
        for i, page in enumerate(pages[:20]): # Increased to 20 pages to catch more structure
            f.write(f"--- Page {i+1} ---\n")
            f.write(page.page_content)
            f.write("\n\n")

if __name__ == "__main__":
    inspect_pdf("constitution_oz.pdf")
