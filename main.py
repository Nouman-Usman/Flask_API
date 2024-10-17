from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser , StrOutputParser
from langchain_groq import ChatGroq
from langchain import hub

# Load environment variables
load_dotenv()
# initialize Pinecone

api_key = os.getenv('PINECONE_API')
legal_index_name = "apna-waqeel"
pc = Pinecone(api_key= api_key)
index = pc.Index(legal_index_name)


# load Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
# initialize vector store and retriever
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vectorstore.as_retriever()


# We use Llama 3.1 by Groq to generate the response
llm = ChatGroq(temperature=0, model="llama3-groq-70b-8192-tool-use-preview")
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a legal assistant assessing the relevance 
    of a retrieved document to a user question related to Pakistani law. If the document contains keywords or information 
    pertinent to the user question, grade it as relevant. The goal is to filter out irrelevant retrievals. \n
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
question = "define law of contract in pakistan"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))