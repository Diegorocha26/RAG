# Notes

NOTE: Use Codex to make this document better

### Encoders
- OpenAI text embedding (small and large)
- Gemini embedding
- Hugging Face all-MiniLM-L6-v2

(TIP): check embedding leaderboards such as: https://huggingface.co/spaces/mteb/leaderboard

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


### Model selection before Evals

Before even running evals, optimizing RAG or even fine-tunning. The best time spent would be chosing the best model possible for a particular use case. Before optimizing anyhthing else, model selection could yield the best ROI in terms of time-spent / quality of the system (responses).

Here are some great reasources that rank models based on certain benchmarks:
- Artificial Analysis: https://artificialanalysis.ai/  
- Vellum: https://www.vellum.ai/llm-leaderboard?utm_source=google&utm_medium=organic  
- Scale (SEAL): https://scale.com/leaderboard  
- LiveBench: https://livebench.ai/#/  
- LMArena: https://arena.ai/leaderboard/  
- HugginFace Spaces: https://huggingface.co/spaces?search=leaderboard  
    - (Plus): also provides leaderboards for embedding models and other specifics (model performance, OCR, speed, industry specific, etc)
