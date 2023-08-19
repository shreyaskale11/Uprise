

#bert 
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import pandas as pd
import os
os.environ['OPENAI_API_KEY'] = ""
print("hello world")

df = pd.read_csv('src\datasets\Amazon Sale Report.csv\Amazon Sale Report.csv',low_memory=False)
print(df.head())
# df to txt for textloader each col space seperated and row in each line 
df[:5].to_csv('src\datasets\Amazon Sale_Report_1.txt', sep=' ', index=False)
loader = TextLoader('src\datasets\Amazon Sale_Report_1.txt')
# print(loader)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(texts)
print("------------",texts[:1000])
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

query = "summaries data and provide insights"
print(qa.run(query))
