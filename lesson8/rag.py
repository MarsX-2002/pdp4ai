from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from law_loader import load_constitution

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY")  # Better to use environment variables
)

# 1. Load and process the document with metadata extraction
def load_and_process_documents(file_path):
    # Use custom constitution loader with metadata
    documents = load_constitution(file_path)
    # No need for splitting - already split by article (modda)
    return documents

# 2. Create a vector store
def create_vector_store(documents):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)

# 3. Create a retriever with metadata filtering support  
def create_retriever(vector_store, k=3):
    return vector_store.as_retriever(search_kwargs={"k": k})

# Add a metadata-aware search function
import re

def metadata_aware_search(query, vector_store, documents, k=3):
    """
    Search with metadata awareness. 
    If query contains specific Bob/Modda reference, filter by metadata first.
    """
    # Check if query contains specific article reference (e.g., "1 bob 3 modda")
    bob_pattern = re.search(r'([IVXLCDMivxlcdm]+)\s+bob', query, re.IGNORECASE)
    modda_pattern = re.search(r'(\d+)[\s-]*modda', query, re.IGNORECASE)
    
    if bob_pattern or modda_pattern:
        # Filter documents by metadata
        filtered_docs = []
        
        for doc in documents:
            match = True
            if bob_pattern:
                bob_query = bob_pattern.group(1).upper()
                if doc.metadata.get('bob', '').upper() != bob_query:
                    match = False
            if modda_pattern:
                modda_query = modda_pattern.group(1)
                if doc.metadata.get('modda', '') != modda_query:
                    match = False
            
            if match:
                filtered_docs.append(doc)
        
        # If we found exact matches, return them
        if filtered_docs:
            return filtered_docs[:k]
    
    # Otherwise, fall back to vector search
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


# 4. Create a QA chain
def create_qa_chain(retriever, llm):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

def main():
    # File path to your PDF
    file_path = "constitution_oz.pdf"
    
    # Load and process documents
    print("Loading and processing documents...")
    documents = load_and_process_documents(file_path)
    print(f"Loaded {len(documents)} articles")
    
    # Create vector store
    print("Creating vector store...")
    vector_store = create_vector_store(documents)
    
    # Example queries
    queries = [
        "Fuqarolar davlat ishlarida qanday ishtirok etishlari mumkin?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Check if this is a specific article query (e.g., "1 bob 3 modda")
        bob_pattern = re.search(r'([IVXLCDMivxlcdm]+)\s+bob', query, re.IGNORECASE)
        modda_pattern = re.search(r'(\d+)[\s-]*modda', query, re.IGNORECASE)
        is_specific_article_query = bool(bob_pattern and modda_pattern)
        
        # Get relevant documents using metadata-aware search
        # Use more results for general queries to improve recall
        k = 3 if is_specific_article_query else 5
        relevant_docs = metadata_aware_search(query, vector_store, documents, k=k)
        
        # Create a custom prompt with the context INCLUDING metadata
        context_parts = []
        for doc in relevant_docs:
            meta = doc.metadata
            # Include metadata in context so LLM knows which article this is
            article_header = f"[{meta.get('bob', 'N/A')} bob {meta.get('modda', 'N/A')}-modda - {meta.get('bob_name', '')}]"
            context_parts.append(f"{article_header}\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Different prompts for specific vs general queries
        if is_specific_article_query:
            prompt = f"""Siz O'zbekiston Konstitutsiyasi bo'yicha maslahatchi botsiz. 

Savol aniq modda haqida so'rayapti. Quyida shu modda matni keltirilgan.

Manbalar:
{context}

Savol: {query}

MUHIM: Agar savol "X bob Y modda nima deyiladi?" yoki shunga o'xshash bo'lsa, manbalardan shu moddaning to'liq matnini javob sifatida bering. "Ma'lumot yo'q" deb javob bermang!

Javob:"""
        else:
            prompt = f"""Siz O'zbekiston Konstitutsiyasi bo'yicha maslahatchi botsiz.

Quyidagi konstitutsiya moddalaridan foydalanib savolga javob bering. 

MUHIM: 
- Agar to'g'ridan-to'g'ri javob bo'lmasa ham, eng yaqin tegishli ma'lumotni tahlil qiling
- Masalan, "yagona fuqarolik" degan ibora "faqat bitta fuqarolik" ma'nosini anglatadi
- Moddalarni diqqat bilan o'qib, mantiqiy xulosa chiqaring

Manbalar:
{context}

Savol: {query}

Javob (moddalardan mantiqiy xulosa chiqaring, agar haqiqatan ham ma'lumot bo'lmasa "Bu mavzu Konstitutsiyada keltirilmagan" deb bildiring):"""
        
        # Get response from LLM
        print("\nGenerating response...")
        response = llm.invoke(prompt)
        
        print("\nAnswer:")
        print(response.content)
        print("\nSources:")
        for i, doc in enumerate(relevant_docs):
            metadata = doc.metadata
            print(f"\nSource {i+1}:")
            print(f"  Bob: {metadata.get('bob', 'N/A')} - {metadata.get('bob_name', 'N/A')}")
            print(f"  Modda: {metadata.get('modda', 'N/A')}")
            print(f"  Content: {doc.page_content[:200]}...")

if __name__ == "__main__":
    main()