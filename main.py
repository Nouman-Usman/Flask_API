import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from typing import List, Dict,TypedDict
from pprint import pprint
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]
class RAGAgent:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('PINECONE_API')
        self.legal_index_name = "lang-graph"
        self.web_search_index_name = "web-search-legal"
        self.pc = Pinecone(api_key=self.api_key)    
        # print(os.getenv('PINECONE_API'))
        # breakpoint()    
        self.index = self.pc.Index(self.legal_index_name)
        self.web_search_index = self.pc.Index(self.web_search_index_name)
        # print("Done")
        # breakpoint()
        self.llm = ChatGroq(temperature=0, model="llama3-groq-70b-8192-tool-use-preview")
        self.vectorstore = None
        self.retriever = None
        self.web_search_tool = TavilySearchResults(k=3)
        self._initialize_vectorstore()
        self._initialize_prompts()

    def _initialize_vectorstore(self):
        # urls = [
        #     "https://lilianweng.github.io/posts/2023-06-23-agent/",
        #     "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        #     "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        # ]
        # docs = [WebBaseLoader(url).load() for url in urls]
        # docs_list = [item for sublist in docs for item in sublist]
        # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        #     chunk_size=250, chunk_overlap=0
        # )
        # doc_splits = text_splitter.split_documents(docs_list)
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        self.vectorstore = PineconeVectorStore(index=self.index, embedding=embeddings)
        # self.vectorstore.add_documents(doc_splits)
        self.retriever = self.vectorstore.as_retriever()
        print("Vectorstore initialized and documents added.")

    def _initialize_prompts(self):
        self.retrieval_grader_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
            of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
             <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["question", "document"],
        )
        self.retrieval_grader = self.retrieval_grader_prompt | self.llm | JsonOutputParser()

        self.generate_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a legal assistant for question-answering tasks in the context of Pakistani law. Use the following pieces of retrieved legal information to answer the query. If you are unsure about the answer, simply state that. Provide well structured answers. |eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {question} 
            Context: {context} 
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question", "context"],
        )
        self.rag_chain = self.generate_prompt | self.llm | StrOutputParser()

        self.hallucination_grader_prompt = PromptTemplate(
            template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
            an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate 
            whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
            single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "documents"],
        )
        self.hallucination_grader = self.hallucination_grader_prompt | self.llm | JsonOutputParser()

        self.answer_grader_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
             <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "question"],
        )
        self.answer_grader = self.answer_grader_prompt | self.llm | JsonOutputParser()

        self.question_router_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
            user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents, 
            prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
            in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
            or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
            no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["question"],
        )
        self.question_router = self.question_router_prompt | self.llm | JsonOutputParser()

    def retrieve(self, state: Dict) -> Dict:
        print("---RETRIEVE---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def generate(self, state: Dict) -> Dict:
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def grade_documents(self, state: Dict) -> Dict:
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score['score']
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def web_search(self, state: Dict) -> Dict:
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents",[])
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]
        return {"documents": documents, "question": question}

    def route_question(self, state: Dict) -> str:
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = self.question_router.invoke({"question": question})  
        if source['datasource'] == 'web_search':
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source['datasource'] == 'vectorstore':
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    def decide_to_generate(self, state: Dict) -> str:
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state["web_search"]
        if web_search == "Yes":
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state: Dict) -> str:
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            score = self.answer_grader.invoke({"question": question,"generation": generation})
            grade = score['score']
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    def build_workflow(self):
        workflow = StateGraph(state_schema=GraphState)
        workflow.add_node("websearch", self.web_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.set_conditional_entry_point(
            self.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        workflow.add_edge("websearch", "generate")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
            },
        )
        return workflow.compile()


    def run(self, question: str):
        app = self.build_workflow()
        inputs = {"question": question}
        for output in app.stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running: {key}:")
        pprint(value["generation"])

if __name__ == "__main__":
    agent = RAGAgent()
    print("Thinking...")
    agent.run("Explain THE CO-OPERATIVE SOCIETIES ACT?")