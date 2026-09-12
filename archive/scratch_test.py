import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import tempfile
print("Importing chromadb...")
import chromadb
print("Importing VectorStore...")
from src.rag.vector_store import VectorStore

print("Creating temporary directory...")
tmp_dir = tempfile.mkdtemp()
print("Temp directory created:", tmp_dir)

try:
    print("Calling chromadb.PersistentClient...")
    client = chromadb.PersistentClient(path=tmp_dir)
    print("PersistentClient created.")
    
    print("Getting or creating collection...")
    collection = client.get_or_create_collection("research_papers")
    print("Collection retrieved/created:", collection)
    
    print("Stopping system...")
    client._system.stop()
    print("System stopped.")
except Exception as e:
    print("Error:", e)
finally:
    import shutil
    try:
        shutil.rmtree(tmp_dir)
        print("Cleaned up temp dir.")
    except Exception as e:
        print("Cleanup error:", e)
print("Done!")
