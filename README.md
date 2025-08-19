# ollama_rag
my implementation of a telegram bot powered by local RAG with ollama and langchain

some jupyter notebooks have some interrupted errors or changes that are not consistent


dependency:
+ ollama
+ langchain
+ unstructured
+ cuda (of course)
+ python / conda / jupyter

vector database:
+ qdrant (or your own choice)

telegram:
+ telebot


V3 updated with long context double RAG using sklearn and hdf5 for easier maintenance

original version using phi4 has surprisingly good performance even with shorter context window, can be used with lower vram

V4 added, using UMAP with nearest neighbor allows much faster retrieval and lower memory use (~95% reduction in my use)
