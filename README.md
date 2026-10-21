
## Documind-V2
```⚡ Instant  🧠 Intelligent  📄 Accurate  🔒 Privacy-friendly ```

```no-login PDF Q&A designed to remove user friction.```

Visit the app: https://documindv2.netlify.app/

Documind V2 is a lightweight Retrieval-Augmented Generation (RAG) application that allows users to chat with their PDFs immediately, without the barriers of account creation, complex dashboards, or paywalls.

# 🛠️ Technical Highlights

Hybrid Retrieval: Combines lexical search (BM25) and semantic search (Vector embeddings via sentence-transformers) for robust context grounding.
Reciprocal Rank Fusion (RRF): Custom implementation to intelligently merge and rank results from both retrieval methods.
Multi-Model Failover: Built-in logic to gracefully handle API rate limits or downtime by routing to backup models.
Frictionless UX: Designed for immediate, single-session use without requiring user authentication.

# 🚀 How to Use

Visit the live application: https://documindv2.netlify.app/

Upload a PDF document.

Start asking questions immediately.

# ⚠️ Project Scope & Limitations

Note: This is an active learning project optimized for conceptual clarity and demonstration purposes. It utilizes in-memory storage and external API dependencies, making it unsuitable for large-scale, multi-user production deployment.
