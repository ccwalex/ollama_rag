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

hardware I run with approximately 20 gb of pdf and some xml for RAG
+ intel 12th gen i7
+ 64 GB ram
+ 2 nvidia geforce 4080s
+ 60gb ssd swap


V3 updated with long context double RAG using sklearn and hdf5 for easier maintenance
