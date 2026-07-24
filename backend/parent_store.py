import os
import pickle

class ParentStore:
    """
    A lightweight, disk-backed cache database using Python pickle serialization.
    Stores parent document texts and coordinates without embeddings,
    used to expand retrieved child chunks back to their complete sections.
    """
    def __init__(self, db_path):
        self.filepath = os.path.join(db_path, "parent_documents.pkl")
        self.parents = {}
        self.load()
        
    def load(self):
        """Loads parent document cache from disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "rb") as f:
                    self.parents = pickle.load(f)
            except Exception as e:
                print(f"[WARNING] ParentStore failed to load from '{self.filepath}': {e}")
                self.parents = {}
        else:
            self.parents = {}
            
    def save(self):
        """Saves parent document cache to disk."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        try:
            with open(self.filepath, "wb") as f:
                pickle.dump(self.parents, f)
        except Exception as e:
            print(f"[ERROR] ParentStore failed to write to '{self.filepath}': {e}")
            
    def add_parents(self, parent_list):
        """Adds a list of parent document dicts to the cache and saves."""
        for p in parent_list:
            self.parents[p["parent_id"]] = p
        self.save()
        
    def get_parent(self, parent_id):
        """Retrieves a single parent document by its unique ID."""
        return self.parents.get(parent_id)
        
    def delete_by_document(self, document_filename):
        """Deletes all cached parent documents matching a specific PDF filename."""
        keys_to_delete = [k for k, v in self.parents.items() if v.get("filename") == document_filename]
        for k in keys_to_delete:
            if k in self.parents:
                del self.parents[k]
        self.save()
        
    def clear(self):
        """Resets the cache and deletes the file on disk."""
        self.parents = {}
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except Exception as e:
                print(f"[WARNING] ParentStore failed to delete '{self.filepath}': {e}")
