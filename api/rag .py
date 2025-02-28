from langchain_core.globals import set_verbose, set_debug
from langchain_community.vectorstores import chroma
from langchain_community.chat_models import ChatOllamaw
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import numpy as np 

set_debug(True)
set_verbose(True)

class Mentor: 
    def __init__(self, llm_model : str = "qwen2.5"):
        self.model = ChatOllamaw(llm_model = llm_model)
        self.chroma_client = chroma
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=100
        )
        self.prompt = ChatPromptTemplate(
            """
            Prompt:

            You are an experienced and knowledgeable mentor, guiding users by answering their questions strictly based on the provided context. Your responses should be clear, accurate, and helpful while maintaining a friendly and supportive tone.
            Instructions:

                Use only the provided context to answer questions. If the answer is not in the context, say, "I don’t have enough information to answer that."
                Provide structured and detailed explanations when necessary.
                Keep responses concise and relevant, avoiding unnecessary information.
                If the question is ambiguous, ask for clarification instead of making assumptions.

            Example Format:

            User Question: What is the process for submitting a proposal?
            Context Provided: "To submit a proposal, visit the online portal, fill out the required form, and upload your document. Proposals are reviewed within two weeks."
            Response: "To submit a proposal, go to the online portal, complete the form, and upload your document. The review process takes approximately two weeks."

            If No Relevant Context is Available:
            "I don’t have enough information to answer that. Could you provide more details or check the provided resources?"
                        """
        )
        self.vector_store = None
        self.retriever = None
        self.chain = None

    def store_embeddings(self):
        print("Storing embeddings")

    def ask(self, query: str):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory="chroma_db", embedding=FastEmbedEmbeddings()
            )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        self.retriever.invoke(query)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        return self.chain.invoke(query)

    def load_embeddings(self, path: str):
        embddings = np.load(path)

