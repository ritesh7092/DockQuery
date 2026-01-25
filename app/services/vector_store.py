import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.models.schemas import Document, DocumentMetadata
from typing import List, Dict, Optional, Any
import numpy as np
import logging
from dataclasses import asdict

# Configure logging
logger = logging.getLogger(__name__)

class VectorStore:
    """
    ChromaDB-based vector store for multimodal document retrieval.
    Supports text, images, tables, and charts with metadata filtering and MMR.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector store with a persistent ChromaDB client.
        
        Args:
            persist_directory: Directory for persistent storage. Defaults to settings.CHROMA_DB_PATH
        """
        self.persist_directory = persist_directory or settings.CHROMA_DB_PATH
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            logger.info(f"ChromaDB client initialized at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}") from e
        
        self.collections: Dict[str, chromadb.Collection] = {}
        
    def initialize_collection(
        self, 
        collection_name: str = "multimodal_rag",
        reset: bool = False
    ) -> chromadb.Collection:
        """
        Initialize or retrieve a collection.
        
        Args:
            collection_name: Name of the collection
            reset: If True, delete existing collection and create new one
            
        Returns:
            ChromaDB collection instance
        """
        try:
            if reset and collection_name in [c.name for c in self.client.list_collections()]:
                logger.info(f"Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name)
            
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            self.collections[collection_name] = collection
            logger.info(f"Collection '{collection_name}' initialized successfully")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to initialize collection '{collection_name}': {e}")
            raise RuntimeError(f"Collection initialization failed: {e}") from e
    
    def add_documents(
        self,
        docs: List[Document],
        embeddings: List[np.ndarray],
        collection_name: str = "multimodal_rag"
    ) -> None:
        """
        Add documents with embeddings to the collection.
        
        Args:
            docs: List of Document objects
            embeddings: List of embedding vectors (np.ndarray)
            collection_name: Target collection name
        """
        if not docs or not embeddings:
            raise ValueError("Documents and embeddings cannot be empty")
        
        if len(docs) != len(embeddings):
            raise ValueError(f"Mismatch: {len(docs)} documents but {len(embeddings)} embeddings")
        
        # Get or initialize collection
        collection = self.collections.get(collection_name)
        if collection is None:
            collection = self.initialize_collection(collection_name)
        
        try:
            # Prepare data for ChromaDB
            ids = [doc.id for doc in docs]
            documents = [doc.content for doc in docs]
            metadatas = []
            
            for doc in docs:
                metadata = {
                    "type": doc.metadata.type,
                    "page": doc.metadata.page,
                    "source": doc.metadata.source,
                }
                if doc.metadata.image_path:
                    metadata["image_path"] = doc.metadata.image_path
                if doc.metadata.bbox:
                    # Convert bbox to ChromaDB-compatible format
                    # Handle both dict and list/tuple formats
                    if isinstance(doc.metadata.bbox, dict):
                        for k, v in doc.metadata.bbox.items():
                            metadata[f"bbox_{k}"] = float(v)
                    elif isinstance(doc.metadata.bbox, (list, tuple)) and len(doc.metadata.bbox) == 4:
                        # Convert [x0, y0, x1, y1] to dict format
                        metadata["bbox_x0"] = float(doc.metadata.bbox[0])
                        metadata["bbox_y0"] = float(doc.metadata.bbox[1])
                        metadata["bbox_x1"] = float(doc.metadata.bbox[2])
                        metadata["bbox_y1"] = float(doc.metadata.bbox[3])
                    else:
                        logger.warning(f"Unexpected bbox format for document {doc.id}: {type(doc.metadata.bbox)}")
                metadatas.append(metadata)
            
            # Convert embeddings to list format
            embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                             for emb in embeddings]
            
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings_list
            )
            logger.info(f"Added {len(docs)} documents to collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise RuntimeError(f"Document insertion failed: {e}") from e
    
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: str = "multimodal_rag"
    ) -> List[Document]:
        """
        Perform similarity search with optional metadata filtering.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filters: Metadata filters (e.g., {"type": "image", "page": 1})
            collection_name: Collection to search
            
        Returns:
            List of Document objects
        """
        collection = self.collections.get(collection_name)
        if collection is None:
            collection = self.initialize_collection(collection_name)
        
        try:
            # Build where clause from filters
            where_clause = self._build_where_clause(filters) if filters else None
            
            # Convert embedding to list
            query_emb_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            results = collection.query(
                query_embeddings=[query_emb_list],
                n_results=k,
                where=where_clause,
                include=["embeddings", "documents", "metadatas", "distances"]
            )
            
            # Convert results to Document objects
            documents = self._results_to_documents(results)
            logger.info(f"Similarity search returned {len(documents)} results")
            return documents
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise RuntimeError(f"Search operation failed: {e}") from e
    
    def hybrid_search(
        self,
        text_query: str,
        k: int = 5,
        collection_name: str = "multimodal_rag"
    ) -> Dict[str, List[Document]]:
        """
        Perform hybrid search across text and visual content.
        Note: This requires a text embedding. Use with EmbeddingService.
        
        Args:
            text_query: Text query string (must be embedded externally)
            k: Number of results per category
            collection_name: Collection to search
            
        Returns:
            Dictionary with keys: "text", "image", "table", "chart"
        """
        # This is a placeholder - actual implementation requires embedding service
        # For now, return structure showing what would be returned
        logger.warning("hybrid_search requires external embedding. Use similarity_search with type filters instead.")
        return {
            "text": [],
            "image": [],
            "table": [],
            "chart": []
        }
    
    def mmr_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        lambda_param: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: str = "multimodal_rag",
        fetch_k: int = 20
    ) -> List[Document]:
        """
        Maximal Marginal Relevance search for diverse results.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            filters: Metadata filters
            collection_name: Collection to search
            fetch_k: Number of candidates to fetch before MMR
            
        Returns:
            List of diverse Document objects
        """
        collection = self.collections.get(collection_name)
        if collection is None:
            collection = self.initialize_collection(collection_name)
        
        try:
            # Fetch more candidates than needed
            where_clause = self._build_where_clause(filters) if filters else None
            query_emb_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            results = collection.query(
                query_embeddings=[query_emb_list],
                n_results=min(fetch_k, k * 4),  # Fetch 4x or fetch_k candidates
                where=where_clause,
                include=["embeddings", "documents", "metadatas", "distances"]
            )
            
            if not results["ids"][0]:
                return []
            
            # Extract embeddings and convert to numpy
            candidate_embeddings = [np.array(emb) for emb in results["embeddings"][0]]
            
            # Apply MMR algorithm
            selected_indices = self._calculate_mmr(
                query_embedding=query_embedding,
                candidate_embeddings=candidate_embeddings,
                k=k,
                lambda_param=lambda_param
            )
            
            # Build documents from selected indices
            documents = []
            for idx in selected_indices:
                doc = self._result_to_document(
                    doc_id=results["ids"][0][idx],
                    content=results["documents"][0][idx],
                    metadata=results["metadatas"][0][idx],
                    embedding=candidate_embeddings[idx]
                )
                documents.append(doc)
            
            logger.info(f"MMR search returned {len(documents)} diverse results")
            return documents
            
        except Exception as e:
            logger.error(f"MMR search failed: {e}")
            raise RuntimeError(f"MMR search operation failed: {e}") from e
    
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection from the database.
        
        Args:
            collection_name: Name of collection to delete
        """
        try:
            self.client.delete_collection(collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            logger.info(f"Collection '{collection_name}' deleted successfully")
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise RuntimeError(f"Collection deletion failed: {e}") from e
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build ChromaDB where clause from filters.
        
        Args:
            filters: Dictionary of metadata filters
            
        Returns:
            ChromaDB where clause
        """
        where = {}
        for key, value in filters.items():
            if isinstance(value, list):
                where[key] = {"$in": value}
            else:
                where[key] = value
        return where
    
    def _calculate_mmr(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        k: int,
        lambda_param: float
    ) -> List[int]:
        """
        Calculate MMR to select diverse results.
        
        Algorithm: Iteratively select documents that maximize:
        λ * similarity(query, doc) - (1-λ) * max(similarity(doc, selected))
        
        Args:
            query_embedding: Query vector
            candidate_embeddings: List of candidate embeddings
            k: Number of results to select
            lambda_param: Balance parameter
            
        Returns:
            List of selected indices
        """
        if not candidate_embeddings:
            return []
        
        # Calculate similarities to query (using cosine similarity)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidate_norms = [emb / np.linalg.norm(emb) for emb in candidate_embeddings]
        query_sims = [np.dot(query_norm, cand_norm) for cand_norm in candidate_norms]
        
        selected_indices = []
        remaining_indices = list(range(len(candidate_embeddings)))
        
        # Select first document (highest similarity to query)
        best_idx = int(np.argmax(query_sims))
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Iteratively select remaining documents
        while len(selected_indices) < k and remaining_indices:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                # Relevance to query
                relevance = query_sims[idx]
                
                # Maximum similarity to already selected documents
                max_sim_to_selected = max(
                    np.dot(candidate_norms[idx], candidate_norms[sel_idx])
                    for sel_idx in selected_indices
                )
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        return selected_indices
    
    def _results_to_documents(self, results: Dict[str, Any]) -> List[Document]:
        """Convert ChromaDB results to Document objects."""
        documents = []
        
        if not results["ids"][0]:
            return documents
        
        for i in range(len(results["ids"][0])):
            doc = self._result_to_document(
                doc_id=results["ids"][0][i],
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                embedding=np.array(results["embeddings"][0][i]) if "embeddings" in results else None
            )
            documents.append(doc)
        
        return documents
    
    def _result_to_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        embedding: Optional[np.ndarray] = None
    ) -> Document:
        """Convert a single ChromaDB result to a Document object."""
        # Extract bbox if present
        bbox = None
        bbox_keys = [k for k in metadata.keys() if k.startswith("bbox_")]
        if bbox_keys:
            bbox = {k.replace("bbox_", ""): metadata[k] for k in bbox_keys}
        
        doc_metadata = DocumentMetadata(
            type=metadata.get("type", "text"),
            page=metadata.get("page", 1),
            source=metadata.get("source", ""),
            image_path=metadata.get("image_path"),
            bbox=bbox
        )
        
        return Document(
            id=doc_id,
            content=content,
            metadata=doc_metadata,
            embedding=embedding
        )

# Global instance
vector_store = VectorStore()
