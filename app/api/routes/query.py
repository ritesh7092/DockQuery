from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store
from app.services.agent import rag_agent

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        # Embed query
        query_embedding = embedding_service.get_text_embedding(request.query)
        
        # Retrieve documents
        results = vector_store.query(query_embedding)
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        context = "\n".join(documents)
        
        # Generate answer
        answer = rag_agent.generate_response(request.query, context)
        
        return QueryResponse(answer=answer, sources=metadatas)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
