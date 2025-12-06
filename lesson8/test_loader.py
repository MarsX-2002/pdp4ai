"""Test script to validate the custom constitution loader."""
from law_loader import load_constitution

def test_loader():
    print("Testing Constitution Loader...")
    documents = load_constitution("constitution_oz.pdf")
    
    print(f"\nTotal articles extracted: {len(documents)}")
    
    # Show first 5 articles
    print("\n=== First 5 Articles ===")
    for i, doc in enumerate(documents[:5]):
        metadata = doc.metadata
        print(f"\nArticle {i+1}:")
        print(f"  Bo'lim: {metadata.get('bolim')} - {metadata.get('bolim_name')}")
        print(f"  Bob: {metadata.get('bob')} - {metadata.get('bob_name')}")
        print(f"  Modda: {metadata.get('modda')}")
        print(f"  Content: {doc.page_content[:150]}...")
    
    # Find a specific article (e.g., Modda 3 from Bob I)
    print("\n=== Looking for Article 3 (1 bob 3 modda) ===")
    for doc in documents:
        if doc.metadata.get('modda') == '3' and doc.metadata.get('bob') == 'I':
            metadata = doc.metadata
            print(f"Found!")
            print(f"  Bo'lim: {metadata.get('bolim')} - {metadata.get('bolim_name')}")
            print(f"  Bob: {metadata.get('bob')} - {metadata.get('bob_name')}")
            print(f"  Modda: {metadata.get('modda')}")
            print(f"  Content: {doc.page_content}")
            break

if __name__ == "__main__":
    test_loader()
