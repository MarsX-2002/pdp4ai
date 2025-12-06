import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class ConstitutionLoader:
    """Custom loader for Constitution of Uzbekistan with hierarchical metadata extraction."""
    
    def __init__(self, file_path):
        self.file_path = file_path
        
    def load(self):
        """Load the PDF and extract documents with metadata."""
        # Load raw pages
        loader = PyPDFLoader(self.file_path)
        pages = loader.load()
        
        # Combine all pages into one text
        full_text = "\n".join([page.page_content for page in pages])
        
        # Extract structured documents
        documents = self._extract_articles(full_text)
        
        return documents
    
    def _extract_articles(self, text):
        """Extract articles with hierarchical metadata (bo'lim -> bob -> modda)."""
        documents = []
        
        # State tracking
        current_bolim = None
        current_bolim_name = None
        current_bob = None
        current_bob_name = None
        
        # Split text into lines
        lines = text.split('\n')
        
        # Patterns
        # Bo'lim pattern: Matches both ' (U+2018) and ' (U+2019) Unicode apostrophes
        bolim_pattern = re.compile(r"^([A-Z]{2,})\s+BO[''\u2018\u2019]LIM[\.\s]*(.*?)$", re.IGNORECASE)
        bob_pattern = re.compile(r'^([IVXLCDM]+)\s+bob[\.\s]*(.*?)$', re.IGNORECASE)
        modda_pattern = re.compile(r'^(\d+)-modda[\.\s]*$')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for Bo'lim (Section)
            bolim_match = bolim_pattern.match(line)
            if bolim_match:
                current_bolim = bolim_match.group(1)
                # Get the section name - may span multiple lines
                name = bolim_match.group(2).strip()
                # If name is incomplete or empty, collect from next lines
                if len(name) < 10 and i + 1 < len(lines):  # Section names are usually longer
                    next_line = lines[i + 1].strip()
                    if next_line and not bob_pattern.match(next_line):
                        name = (name + " " + next_line).strip()
                current_bolim_name = name if name else ""
                i += 1
                continue
            
            # Check for Bob (Chapter)
            bob_match = bob_pattern.match(line)
            if bob_match:
                current_bob = bob_match.group(1)
                # Try to get the chapter name from the same or next line
                name = bob_match.group(2).strip()
                if not name and i + 1 < len(lines):
                    name = lines[i + 1].strip()
                current_bob_name = name if name else ""
                i += 1
                continue
            
            # Check for Modda (Article)
            modda_match = modda_pattern.match(line)
            if modda_match:
                modda_number = modda_match.group(1)
                
                # Extract article content (until next modda or section/chapter)
                article_content = []
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    
                    # Stop if we hit another modda, bob, or bo'lim
                    if (modda_pattern.match(next_line) or 
                        bob_pattern.match(next_line) or 
                        bolim_pattern.match(next_line)):
                        break
                    
                    if next_line:  # Only add non-empty lines
                        article_content.append(next_line)
                    i += 1
                
                # Create document with metadata
                content = " ".join(article_content)
                if content:  # Only add if there's content
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": self.file_path,
                            "bolim": current_bolim if current_bolim else "N/A",
                            "bolim_name": current_bolim_name if current_bolim_name else "N/A",
                            "bob": current_bob if current_bob else "N/A",
                            "bob_name": current_bob_name if current_bob_name else "N/A",
                            "modda": modda_number,
                            "type": "article"
                        }
                    )
                    documents.append(doc)
                continue
            
            i += 1
        
        return documents


def load_constitution(file_path):
    """Helper function to load constitution with metadata."""
    loader = ConstitutionLoader(file_path)
    return loader.load()


# test = load_constitution("constitution_oz.pdf")
# print(test[0])