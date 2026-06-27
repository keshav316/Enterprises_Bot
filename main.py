"""
=========================================================
Enterprise Knowledge Assistant - RAG Application

Developer : Keshav Gupta

LLM        : Llama 3 (Ollama)
Vector DB  : ChromaDB
Framework  : LangChain
Interface  : Streamlit

Features:
- Upload enterprise documents
- PDF, DOCX, XLSX, TXT support
- Document indexing
- Semantic retrieval
- Streaming LLM answering
- Source citation

=========================================================
"""


import os
import streamlit as st


# ==============================
# LangChain Imports
# ==============================

from langchain_groq import ChatGroq
from dotenv import load_dotenv

from langchain_chroma import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import (
    RunnablePassthrough
)

from langchain_core.output_parsers import (
    StrOutputParser
)


load_dotenv()

# ==============================
# Enterprise Bot Configuration
# ==============================


# ==============================
# Groq LLM Configuration
# ==============================


groq_api_key = os.getenv(
    "GROQ_API"
)


if not groq_api_key:

    raise ValueError(
        "GROQ_API_KEY not found in .env file"
    )



llm = ChatGroq(

    api_key=groq_api_key,

    model="llama-3.1-8b-instant",

    temperature=0.3,

    streaming=True

)



# ==============================
# Embedding Model
# ==============================


embedding_model = HuggingFaceEmbeddings(

    model_name=
    "sentence-transformers/all-MiniLM-L6-v2"

)



# ==============================
# Document Loader
# ==============================


def load_documents(file_path):


    if file_path.endswith(".pdf"):

        loader = PyPDFLoader(file_path)


    elif file_path.endswith(".docx"):

        loader = Docx2txtLoader(file_path)


    elif file_path.endswith(".xlsx"):

        loader = UnstructuredExcelLoader(file_path)


    elif file_path.endswith(".txt"):

        loader = TextLoader(file_path)


    else:

        raise Exception(
            "Unsupported file type"
        )


    return loader.load()



# ==============================
# Create Chroma Database
# ==============================


def create_vector_database(documents):


    splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=150

    )


    chunks = splitter.split_documents(

        documents

    )


    vector_database = Chroma.from_documents(

        documents=chunks,

        embedding=embedding_model,

        persist_directory="./chroma_db"

    )


    return vector_database



# ==============================
# Create Streaming RAG Pipeline
# ==============================


def create_rag_chain(vector_database):


    retriever = vector_database.as_retriever(

        search_kwargs={
            "k":5
        }

    )


    prompt = PromptTemplate(

        template="""

You are Enterprise Knowledge Assistant.


Answer the question only using the provided context.


Rules:

1. Do not hallucinate.

2. If information is unavailable say:

"I could not find this information in company documents."


3. Give professional enterprise answers.


Context:

{context}


Question:

{question}


Answer:

""",

        input_variables=[

            "context",

            "question"

        ]

    )



    def format_docs(docs):

        return "\n\n".join(

            doc.page_content

            for doc in docs

        )



    rag_chain = (

        {

            "context":
            retriever | format_docs,


            "question":
            RunnablePassthrough()

        }

        |

        prompt

        |

        llm

        |

        StrOutputParser()

    )


    return rag_chain, retriever



# ==============================
# Streamlit Application
# ==============================


def main():


    st.set_page_config(

        page_title=
        "Enterprise Knowledge Assistant",

        layout="wide"

    )


    st.title(
        "🏢 Enterprise Knowledge Assistant"
    )


    st.markdown(

    """

Upload company documents and ask questions.

Supported formats:

- PDF
- DOCX
- XLSX
- TXT


Powered by:

Llama + LangChain + ChromaDB


Developer:

Keshav Gupta

"""

    )



    uploaded_files = st.file_uploader(

        "Upload Enterprise Documents",

        type=[

            "pdf",

            "docx",

            "xlsx",

            "txt"

        ],

        accept_multiple_files=True

    )



    if uploaded_files:


        all_documents=[]


        os.makedirs(

            "data",

            exist_ok=True

        )


        for file in uploaded_files:


            file_path = (

                "data/" + file.name

            )


            with open(

                file_path,

                "wb"

            ) as f:


                f.write(

                    file.getbuffer()

                )


            docs = load_documents(

                file_path

            )


            all_documents.extend(

                docs

            )



        if st.button(

            "Create Knowledge Base"

        ):


            with st.spinner(

                "Processing documents..."

            ):


                vector_db = create_vector_database(

                    all_documents

                )


                st.session_state.qa, st.session_state.retriever = create_rag_chain(

                    vector_db

                )


            st.success(

                "Knowledge base created successfully!"

            )



    # ==============================
    # Chat Interface
    # ==============================


    if "qa" in st.session_state:


        question = st.chat_input(

            "Ask about company documents..."

        )



        if question:


            with st.chat_message("user"):

                st.write(question)



            with st.chat_message("assistant"):


                response_box = st.empty()


                complete_answer = ""


                # TRUE TOKEN STREAMING

                for token in st.session_state.qa.stream(

                    question

                ):


                    complete_answer += token


                    response_box.markdown(

                        complete_answer + "▌"

                    )


                response_box.markdown(

                    complete_answer

                )



            




# ==============================
# Run Application
# ==============================


if __name__ == "__main__":

    main()