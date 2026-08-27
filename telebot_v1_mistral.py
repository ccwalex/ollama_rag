from telegram.ext import Updater, MessageHandler

import os



# Replace 'YOUR_BOT_TOKEN' with the API token obtained from BotFather
BOT_TOKEN = 'Insert bot token'
TELEGRAM_CHAT_ID = 'insert telegram chat id'

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import faiss
FAISS_PATH = "faiss"



# Initialize LLM
llm = Ollama(model="mistral-nemo")

# Initialize embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")


from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(path='qdrant')
print('done')

vector_store = QdrantVectorStore(client=client, collection_name='nomic_multicore', embedding=embeddings, )
vector_retriever = vector_store.as_retriever()
# Set up chain
prompt = ChatPromptTemplate.from_template("""Answer the following question based on the provided context and previous conversations, you may use online sources when requested and you should quote the online sources that you used:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# Set up retiever
retrieval_chain = create_retrieval_chain(vector_retriever, document_chain)

def llm_reply(message):


    # Invoke using RAG
    response = retrieval_chain.invoke({"input": message})
    
    return response['answer']


def respond_to_user(message):
    #print(message)
    response = llm_reply(message = message.text)
    return response


import telebot

bot = telebot.TeleBot(BOT_TOKEN) 

@bot.message_handler(func=lambda message: True)
def reply(message):
	bot.reply_to(message, respond_to_user(message))


bot.infinity_polling()
