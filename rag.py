import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dataLoading import vectorstore  # This now loads cached vectorstore!
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


def initialize_hf_llm():
    base_llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.1,
        max_new_tokens=512,
    )

    llm = ChatHuggingFace(llm=base_llm)
    return llm


llm = initialize_hf_llm()


def create_rag_chain(vectorstore, llm):
    prompt_template = """
You are a helpful legal assistant for Sierra Leone citizens. Your job is to explain laws and policies in simple, everyday language that anyone can understand.

Use the following legal documents to answer the question:

{context}

Question: {question}

Instructions:
- Explain in simple English, as if talking to someone with no legal background
- Break down any legal terms in parentheses (e.g., "plaintiff" means "the person suing")
- Use everyday examples when helpful
- If citing a specific law, mention it clearly (e.g., "According to the Constitution Act 1991, Section 5...")
- Be conversational and friendly, not formal
- If you're not sure or the documents don't contain the answer, say "I don't have enough information to answer this confidently"

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain


# Create the RAG chain (vectorstore is already cached!)
rag_chain = create_rag_chain(vectorstore, llm)


def format_source_display(doc, index):
    """Format source documents in a user-friendly way"""
    metadata = doc.metadata

    source = metadata.get('source', 'Unknown source')
    page = metadata.get('page', 'N/A')

    if 'data/pdfs/' in source:
        source = source.replace('data/pdfs/', '').replace('.pdf', '')

    print(f"\n📄 Source {index}:")
    print(f"   Document: {source}")
    if page != 'N/A':
        print(f"   Page: {page + 1}")
    print(f"   Preview: {doc.page_content[:250]}...")
    print()


def is_relevant_source(doc_content, question, min_relevance=0.3):
    """Basic relevance check"""
    question_keywords = set(question.lower().split())
    content_words = set(doc_content.lower().split())

    overlap = len(question_keywords & content_words)
    if len(question_keywords) == 0:
        return False

    relevance_score = overlap / len(question_keywords)
    return relevance_score >= min_relevance


def query_rag_system(rag_chain, question):
    """Query the RAG system with improved filtering"""

    print("\n" + "=" * 80)
    print("🔍 Searching Sierra Leone legal documents...")
    print("=" * 80)

    try:
        result = rag_chain({"query": question})

        print(f"\n❓ Your Question:")
        print(f"   {question}")
        print(f"\n💡 Answer:")
        print(f"   {result['result']}")

        if result.get('source_documents'):
            relevant_sources = [
                doc for doc in result['source_documents']
                if is_relevant_source(doc.page_content, question)
            ]

            if relevant_sources:
                print(f"\n📚 This answer is based on {len(relevant_sources)} source(s):")
                print("-" * 80)

                for i, doc in enumerate(relevant_sources, 1):
                    format_source_display(doc, i)
            else:
                print(f"\n📚 Sources found but showing top 3 most relevant:")
                print("-" * 80)
                for i, doc in enumerate(result['source_documents'][:3], 1):
                    format_source_display(doc, i)
        else:
            print("\n⚠️  No specific sources were found for this answer.")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please try rephrasing your question or contact support.")


def show_welcome_message():
    """Display welcome message and example questions"""
    print("\n" + "=" * 80)
    print("🇸🇱 SIERRA LEONE LEGAL ASSISTANT")
    print("=" * 80)
    print("\nWelcome! I can help you understand Sierra Leone's laws and policies")
    print("in simple, everyday language.\n")
    print("Example questions you can ask:")
    print("  • What are my rights if I'm arrested?")
    print("  • How do I register a business in Sierra Leone?")
    print("  • What does the Constitution say about freedom of speech?")
    print("  • What are the requirements for voting?")
    print("\nType 'exit' or 'quit' to leave.\n")
    print("=" * 80)


if __name__ == "__main__":
    show_welcome_message()

    while True:
        question = input("\n💬 Ask your question (or type 'exit' to quit): ").strip()

        if not question:
            print("⚠️  Please enter a question.")
            continue

        if question.lower() in ["exit", "quit", "q"]:
            print("\n👋 Thank you for using Sierra Leone Legal Assistant. Goodbye!")
            break

        query_rag_system(rag_chain, question)