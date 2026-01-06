"""
RAG Service - Retrieval Augmented Generation
Handles document loading, embedding, retrieval, and generation.
"""
import os
from pathlib import Path
from typing import Optional
import hashlib

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config


class RAGService:
    """
    Simple RAG implementation without LangChain.
    Demonstrates core RAG concepts:
    - Document loading and chunking
    - Embedding generation
    - Vector storage and retrieval
    - Context-augmented generation
    """
    
    def __init__(self):
        self.kb_path = config.BASE_DIR / "data" / "knowledge_base"
        self.chroma_path = config.CHROMA_PERSIST_DIR
        
        # Initialize embedding model (runs locally)
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"description": "VitalAI medical knowledge base"}
        )
        
        print(f"RAG Service initialized. Collection has {self.collection.count()} documents.")
    
    def load_knowledge_base(self, force_reload: bool = False) -> int:
        """
        Load all markdown documents from the knowledge base directory.
        Returns number of documents loaded.
        """
        if not force_reload and self.collection.count() > 0:
            print(f"Knowledge base already loaded ({self.collection.count()} chunks). Use force_reload=True to reload.")
            return self.collection.count()
        
        # Clear existing if force reload
        if force_reload and self.collection.count() > 0:
            print("Clearing existing knowledge base...")
            # Delete all documents
            all_ids = self.collection.get()['ids']
            if all_ids:
                self.collection.delete(ids=all_ids)
        
        documents = []
        metadatas = []
        ids = []
        
        # Walk through knowledge base directory
        for md_file in self.kb_path.rglob("*.md"):
            chunks = self._load_and_chunk_file(md_file)
            
            for i, chunk in enumerate(chunks):
                # Create unique ID based on file path and chunk index
                doc_id = hashlib.md5(f"{md_file}_{i}".encode()).hexdigest()[:16]
                
                documents.append(chunk['content'])
                metadatas.append({
                    'source': str(md_file.relative_to(self.kb_path)),
                    'title': chunk['title'],
                    'category': md_file.parent.name,
                    'chunk_index': i
                })
                ids.append(doc_id)
        
        if not documents:
            print("No documents found in knowledge base!")
            return 0
        
        # Generate embeddings and add to collection
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedder.encode(documents).tolist()
        
        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Loaded {len(documents)} chunks from {len(list(self.kb_path.rglob('*.md')))} files.")
        return len(documents)
    
    def _load_and_chunk_file(self, file_path: Path) -> list[dict]:
        """
        Load a markdown file and split into chunks.
        Splits on headers (##) to keep related content together.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get document title (first # header)
        lines = content.split('\n')
        doc_title = lines[0].replace('#', '').strip() if lines else file_path.stem
        
        # Split by ## headers
        chunks = []
        current_chunk = []
        current_title = doc_title
        
        for line in lines:
            if line.startswith('## '):
                # Save previous chunk if it has content
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk).strip()
                    if len(chunk_content) > 50:  # Minimum chunk size
                        chunks.append({
                            'title': current_title,
                            'content': chunk_content
                        })
                
                # Start new chunk
                current_title = f"{doc_title} - {line.replace('##', '').strip()}"
                current_chunk = [line]
            else:
                current_chunk.append(line)
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk).strip()
            if len(chunk_content) > 50:
                chunks.append({
                    'title': current_title,
                    'content': chunk_content
                })
        
        # If no chunks created (no ## headers), use whole document
        if not chunks:
            chunks.append({
                'title': doc_title,
                'content': content
            })
        
        return chunks
    
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search the knowledge base for relevant documents.
        Returns list of documents with their metadata and relevance scores.
        """
        if self.collection.count() == 0:
            print("Warning: Knowledge base is empty. Run load_knowledge_base() first.")
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return documents
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """
        Get formatted context string for a query.
        Used to augment LLM prompts.
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            source = doc['metadata'].get('source', 'Unknown')
            title = doc['metadata'].get('title', 'Untitled')
            context_parts.append(f"[Source {i}: {title}]\n{doc['content']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_stats(self) -> dict:
        """Get statistics about the knowledge base."""
        return {
            'total_chunks': self.collection.count(),
            'embedding_model': config.EMBEDDING_MODEL,
            'embedding_dimension': config.EMBEDDING_DIMENSION,
            'knowledge_base_path': str(self.kb_path)
        }


# Global instance for easy access
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create the global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def init_knowledge_base(force_reload: bool = False) -> int:
    """Initialize the knowledge base. Called on app startup."""
    rag = get_rag_service()
    return rag.load_knowledge_base(force_reload=force_reload)

