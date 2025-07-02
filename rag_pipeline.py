from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.schema.retriever import BaseRetriever
from typing import List
import asyncio

class HybridRetriever(BaseRetriever):
    def __init__(self, base_retriever, k=4, keywords=None):
        object.__setattr__(self, 'base_retriever', base_retriever)
        object.__setattr__(self, 'k', k)
        object.__setattr__(self, 'keywords', keywords or [])
        object.__setattr__(self, 'tags', [])
        object.__setattr__(self, 'metadata', {})

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.base_retriever.get_relevant_documents(query)
        if not self.keywords:
            return results[:self.k]

        keyword_filtered = [
            doc for doc in results
            if any(kw.lower() in doc.page_content.lower() for kw in self.keywords)
        ]

        combined = keyword_filtered[:self.k]
        if len(combined) < self.k:
            extras = [doc for doc in results if doc not in combined]
            combined.extend(extras[:self.k - len(combined)])
        return combined

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return await asyncio.to_thread(self.get_relevant_documents, query)

def rag_pipeline(text, model_choice, temperature, max_tokens, hybrid=True, return_chunks=False, highlight_mode=False):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", " "],
        keep_separator=True,
        add_start_index=True
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 8})

    keywords = list(set(word.lower() for chunk in chunks for word in chunk.split() if len(word) > 4))

    if hybrid:
        retriever = HybridRetriever(base_retriever=retriever, k=4, keywords=keywords)

    llm = ChatOpenAI(model_name=model_choice, temperature=temperature, max_tokens=max_tokens)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    if return_chunks:
        return qa_chain, retriever, chunks
    return qa_chain, retriever
