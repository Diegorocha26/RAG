# Notes

NOTE: Use Codex to make this document better

### Encoders
- OpenAI text embedding (small and large)
- Gemini embedding
- Hugging Face all-MiniLM-L6-v2

### Vectorstores
- Open-Source
    - Chroma
    - Qdrant
    - FAISS (in-memory)

- Paid (scalable)
    - Pinecone
    - Weaviate
    
- Mainstream DBs
    - Postgres
    - Mongo
    - Elastic


## TIPs

### 10 RAG Advances Techniques

1. Chunking R&D: experiment with chunking strategies
2. Encoder R&D: select the best Encoder model based on a test set
3. Improve Prompts: general content, the current date, relevant context, and history
4. Document pre-processing: use an LLM to make the chunks and/or text for encoding
5. Query rewriting: use an LLM to convert the user's question into a better RAG query
6. Query expanding: use an LLM to turn the question into multiple RAG queries
7. Re-ranking: use an LLM to sub-select from RAG results
8. Hierarchical: use an LLM to summarize at multiple levels
9. Graph RAG: retrieve content closely related to similar documents
10. Agentic RAG: use Agents for retrieval, combining with Memory and Tools such as SQL
    - Base use case: SQL queries for number (organized data) extraction and Vector Retrieval for word (unorganized data) extraction