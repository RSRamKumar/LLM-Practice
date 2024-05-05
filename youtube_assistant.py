"""
LLM Application Youtube Questioning
"""


from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# pip install -U langchain-community
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    YoutubeLoader,
)
from langchain_community.vectorstores import FAISS  # pip install faiss-cpu
from langchain_openai import (
    OpenAI,  # pip install -U langchain-openai
    OpenAIEmbeddings,
)

load_dotenv()
embeddings = OpenAIEmbeddings()


def create_vector_db_from_youtube_video_url(video_url: str) -> FAISS:
    """
    Method to create vector database
    by extracting the youtube transcript
    """
    transcript = YoutubeLoader.from_youtube_url(video_url).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    return FAISS.from_documents(docs, embeddings)


def get_response_from_query(vector_db: FAISS, query: str, k=4):
    """
    gpt-3.5-turbo-instruct handles up to 4097 tokens.
    Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    https://platform.openai.com/docs/models/gpt-3-5-turbo
    """

    documents = vector_db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in documents])

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        Help me to answer some question based on the video's transcript.
        By understanding the youtube transcript {docs}, please answer 
        the question {question}
    
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = prompt | llm

    response = chain.invoke({"question": query, "docs": docs_page_content})
    response = response.replace("\n", "")
    return response, documents


if __name__ == "__main__":
    vdb = create_vector_db_from_youtube_video_url(
        "https://youtu.be/vFRkOk3NYLg?list=PLP8GkvaIxJP0zDf2TqlGkqlBOjIuzm_BJ"
    )
    QUESTION_TO_ASK = "what do the podcast discuss about pydantic?"
    llm_response, source_docs = get_response_from_query(vdb, QUESTION_TO_ASK)
    print(llm_response)
    print("*" * 100)
    print(source_docs)
