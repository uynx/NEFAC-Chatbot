import logging
import json
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from vector.load import vector_store
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ============================================================================
# NEW CLEAN IMPLEMENTATION
# ============================================================================

def generate_vector_queries(query: str, chat_history: list) -> list:
    """
    Generate 5 queries specifically formatted for the vector store based on user question and chat history.
    
    Args:
        query (str): The user's input query
        chat_history (list): List of previous messages in the conversation
    
    Returns:
        list: List of 5 query strings optimized for vector search
    """
    
    query_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for the New England First Amendment Coalition (NEFAC). 
        Your task is to generate exactly 5 search queries that will be used to search through a vector database 
        containing YouTube video transcripts, summaries, and documents related to NEFAC's work.

        Given the user's question, create 5 different search queries that:
        1. Address the DIRECT question being asked
        2. Explore BROADER THEMES and CONTEXT around the topic that would enhance the answer
        3. Look for RELATED CONCEPTS, BACKGROUND INFORMATION, and FOUNDATIONAL KNOWLEDGE
        4. Consider PRACTICAL APPLICATIONS, EXAMPLES, and CASE STUDIES related to the topic
        5. Search for CHALLENGES, SOLUTIONS, or ALTERNATIVE PERSPECTIVES on the subject

        Make these queries ABSTRACT and COMPREHENSIVE - think about what information would make you give the BEST possible answer, even if it's not directly mentioned in the question.

        IMPORTANT: Be creative and expansive in your search. Consider:
        - Historical context and evolution of the topic
        - Legal frameworks and precedents
        - Best practices and methodologies
        - Common challenges and innovative solutions
        - Cross-cutting themes that might illuminate the topic
        - Expert perspectives and professional advice
        - Real-world examples and case studies

        For topics related to:
        - FOI/Public Records: Include queries about access challenges, legal precedents, best practices, enforcement, litigation, delays, exemptions, appeals
        - First Amendment: Include constitutional principles, case law, practical applications, violations, protections, limits, interpretations
        - Journalism/Media: Include ethics, techniques, legal protections, investigations, sources, verification, storytelling
        - Government Transparency: Include accountability, oversight, public participation, barriers, reform, democracy, citizen engagement
        - Data/Research: Include methodology, accuracy, verification, sources, analysis, presentation, ethics

        Return ONLY a JSON array of exactly 5 strings. No other text.

        Example format:
        ["direct query about the topic", "broader contextual query", "foundational knowledge query", "practical applications query", "challenges and solutions query"]"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Generate 5 vector search queries for: {question}")
    ])
    
    chain = query_generation_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3) | StrOutputParser()
    
    try:
        input_data = {
            "question": query,
            "chat_history": chat_history
        }
        result = chain.invoke(input_data)
        queries = json.loads(result)
        
        if not isinstance(queries, list) or len(queries) != 5:
            logger.warning(f"Query generation returned invalid format: {result}")
            # Fallback to simple variations of the original query
            return [query] * 5
            
        logger.info(f"Generated vector queries: {queries}")
        return queries
        
    except Exception as e:
        logger.error(f"Error generating vector queries: {e}")
        # Fallback to simple variations of the original query
        return [query] * 5

def retrieve_chunks_from_queries(queries: list, k_per_query: int = 3) -> list:
    """
    Retrieve document chunks from the vector store for each query.
    
    Args:
        queries (list): List of query strings
        k_per_query (int): Number of documents to retrieve per query
    
    Returns:
        list: List of unique document chunks with metadata
    """
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": k_per_query}
        )
        
        all_chunks = []
        seen_chunk_ids = set()
        
        for query in queries:
            logger.info(f"Searching vector store with query: {query}")
            docs = retriever.invoke(query)
            
            for doc in docs:
                # Create a unique identifier for this chunk (allows multiple timestamps from same video)
                chunk_id = f"{doc.metadata.get('title', 'unknown')}:{doc.metadata.get('page', '0')}:{hash(doc.page_content[:100])}"
                
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    all_chunks.append(doc)
                    timestamp_info = doc.metadata.get('page', 'unknown')
                    logger.info(f"Retrieved chunk from: {doc.metadata.get('title', 'unknown')} at {timestamp_info} seconds")
        
        logger.info(f"Retrieved {len(all_chunks)} unique chunks from {len(queries)} queries")
        return all_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return []

def generate_response_with_sources(query: str, chat_history: list, chunks: list) -> dict:
    """
    Generate a response based on retrieved chunks and only include sources that were actually used.
    
    Args:
        query (str): The user's original query
        chat_history (list): Conversation history
        chunks (list): Retrieved document chunks
    
    Returns:
        dict: Response with answer and only relevant sources
    """
    
    if not chunks:
        return {
            "answer": "I don't have enough information in the NEFAC database to answer this question.",
            "sources": []
        }
    
    # Format chunks into context with clear source identifiers
    context_parts = []
    chunk_metadata = []
    
    for i, chunk in enumerate(chunks):
        metadata = chunk.metadata
        title = metadata.get('title', 'Unknown Source')
        source_url = metadata.get('source', '')
        doc_type = metadata.get('type', 'unknown')
        timestamp = metadata.get('page', None)
        
        # Format timestamp for display
        timestamp_display = f" (at {timestamp} seconds)" if timestamp and doc_type == 'youtube' else ""
        
        context_parts.append(f"[Source {i+1}] {title}{timestamp_display}\n{chunk.page_content}")
        
        # Store metadata for potential source creation
        chunk_metadata.append({
            "source_id": i+1,
            "title": title,
            "type": doc_type,
            "link": f"{source_url}&t={timestamp}s" if doc_type == 'youtube' and timestamp else source_url,
            "timestamp_seconds": timestamp if doc_type == 'youtube' else None,
            "summary": metadata.get('summary', None)
        })
    
    context = "\n\n".join(context_parts)
    
    # Generate response with explicit source tracking
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for the New England First Amendment Coalition (NEFAC).

        Based on the provided context from NEFAC's database, answer the user's question accurately and helpfully.

        RULES:
        1. Use the information provided in the context to answer the question
        2. If the context contains relevant information, provide a helpful answer based on it
        3. If the context doesn't contain enough information, say so clearly
        4. When referencing information, mention the source (e.g., "According to the Data Cleaning 101 video")
        5. Be conversational and helpful while staying accurate to the context

        After your answer, on a new line, write "SOURCES_USED:" followed by the source numbers you referenced in your answer (e.g., "SOURCES_USED: 1, 3, 5"). If you found relevant information in the context, list those sources. If you couldn't find relevant information, write "SOURCES_USED: none".

        Context from NEFAC database:
        {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    chain = response_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2) | StrOutputParser()
    
    try:
        input_data = {
            "question": query,
            "chat_history": chat_history,
            "context": context
        }
        
        full_response = chain.invoke(input_data)
        
        # Parse the response to separate answer and used sources
        if "SOURCES_USED:" in full_response:
            parts = full_response.split("SOURCES_USED:")
            answer = parts[0].strip()
            sources_used_text = parts[1].strip()
        else:
            answer = full_response.strip()
            sources_used_text = "none"
        
        # Determine which sources to include
        relevant_sources = []
        
        # Check if the answer indicates insufficient information
        insufficient_info_phrases = [
            "I don't have enough information",
            "don't have enough information",
            "insufficient information",
            "not enough information",
            "cannot answer",
            "can't answer"
        ]
        
        answer_indicates_insufficient = any(phrase in answer.lower() for phrase in insufficient_info_phrases)
        
        if not answer_indicates_insufficient and sources_used_text != "none":
            try:
                # Parse source numbers from the response
                if sources_used_text.lower() != "none":
                    source_numbers = []
                    for part in sources_used_text.replace(',', ' ').split():
                        try:
                            num = int(part.strip())
                            if 1 <= num <= len(chunk_metadata):
                                source_numbers.append(num)
                        except ValueError:
                            continue
                    
                    # Include only the sources that were actually used
                    for source_num in source_numbers:
                        relevant_sources.append(chunk_metadata[source_num - 1])
                        
            except Exception as e:
                logger.warning(f"Error parsing sources used: {e}")
                # If parsing fails and answer seems to have content, include no sources to be safe
                relevant_sources = []
        
        logger.info(f"Generated response with {len(relevant_sources)} relevant sources out of {len(chunks)} total chunks")
        
        return {
            "answer": answer,
            "sources": relevant_sources
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "answer": "An error occurred while generating the response.",
            "sources": []
        }

def query_nefac_database_new(query: str, chat_history: list, session_id: str = "abc123") -> dict:
    """
    Main function implementing the new clean approach:
    1. Generate 5 vector store queries
    2. Retrieve chunks from vector store
    3. Generate response based only on retrieved information
    4. Return response with source links
    
    Args:
        query (str): The user's input query
        chat_history (list): List of previous messages in the conversation
        session_id (str): Session ID for chat history management
    
    Returns:
        dict: Dictionary containing the answer and list of sources
    """
    try:
        logger.info(f"Processing query: {query}")
        
        # Step 1: Generate 5 vector store queries
        vector_queries = generate_vector_queries(query, chat_history)
        
        # Step 2: Retrieve chunks from vector store
        chunks = retrieve_chunks_from_queries(vector_queries, k_per_query=5)
        
        # Step 3: Generate response with sources
        result = generate_response_with_sources(query, chat_history, chunks)
        
        logger.info(f"Successfully processed query with {len(result['sources'])} sources")
        return result
        
    except Exception as e:
        logger.error(f"Error in query_nefac_database_new: {e}")
        return {
            "answer": "An error occurred while processing your query.",
            "sources": []
        }

# ============================================================================
# LEGACY IMPLEMENTATIONS (keeping for now as requested)
# ============================================================================

# Define contextualize_q_chain to rephrase query based on chat history
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are an assistant for the New England First Amendment Coalition (NEFAC). You are currently working in a RAG pipeline.
        Given the conversation history and the latest user query, formulate one question that captures the user's intent in a simple way.
        Ensure the question is clear, specific, and relevant to NEFAC's focus areas (e.g., transparency, public access, First Amendment rights).
        If the query is vague, infer the most likely intent based on NEFAC's mission.
        Do not answer the question, just rephrase it.
        
        Examples:
        Query: "Tell me about the fight for immediate access"
        Rephrased: "What is NEFAC's stance on the fight for immediate access to public records?"
        
        Query: "What's NEFAC doing about free speech?"
        Rephrased: "What initiatives is NEFAC undertaking to protect free speech?"
        
        Query: "Why is transparency important?"
        Rephrased: "Why does NEFAC emphasize the importance of government transparency?"
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1) | StrOutputParser()

# Define multi-query chain to generate query variations
multi_query_prompt = ChatPromptTemplate.from_template(
    """
    You are an assistant for the New England First Amendment Coalition (NEFAC).
    Given a user query, generate three alternative questions that capture different aspects of the user's intent, relevant to NEFAC's focus (e.g., transparency, public access, First Amendment).
    These questions should be clear, specific, and suitable for querying a database of YouTube video transcripts and summaries.
    Return the questions as a JSON list.
    
    Example:
    Query: "What is NEFAC's stance on the fight for immediate access?"
    Alternative Questions:
    [
        "What initiatives does NEFAC support for immediate access to public records?",
        "How does NEFAC advocate for reducing delays in public record access?",
        "What legal arguments does NEFAC make for immediate access to government documents?"
    ]
    
    Query: {query}
    Alternative Questions:
    """
)
multi_query_chain = multi_query_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3) | StrOutputParser() | (lambda x: json.loads(x))

# Format retrieved documents into context and source list (LEGACY - SUPPORTS MULTIPLE TIMESTAMPS)
def format_context_and_sources(docs):
    """
    Format a list of Document objects into a context string and a list of source metadata.
    NOW SUPPORTS MULTIPLE ENTRIES FOR SAME TITLE WITH DIFFERENT TIMESTAMPS.
    
    Args:
        docs (list): List of Document objects with page_content and metadata.
    
    Returns:
        tuple: (context string, list of source dictionaries)
    """
    try:
        formatted_context = []
        sources = []
        seen_chunk_ids = set()  # Changed from seen_titles to allow multiple timestamps
        
        for doc in docs:
            if not hasattr(doc, 'metadata') or 'title' not in doc.metadata:
                logger.warning(f"Document missing metadata: {doc}")
                continue
            
            title = doc.metadata['title']
            content = doc.page_content
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown source')
            doc_type = metadata.get('type', 'Unknown type')
            timestamp = metadata.get('page', None)
            
            # Create unique identifier that includes content hash to allow multiple timestamps
            chunk_id = f"{title}:{timestamp}:{hash(content[:100])}"
            
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            
            timestamp_display = f" (at {timestamp} seconds)" if doc_type == 'youtube' and timestamp else ""
            
            formatted_context.append(
                f"Source: {title}{timestamp_display}\n"
                f"Type: {doc_type}\n"
                f"Content: {content}\n"
                f"Link: {source}\n"
            )
            sources.append({
                "title": title,
                "type": doc_type,
                "link": f"{source}&t={timestamp}s" if doc_type == 'youtube' and timestamp else source,
                "timestamp_seconds": timestamp if doc_type == 'youtube' else None,
                "summary": metadata.get('summary', None)
            })
            # Log document metadata for debugging
            logger.info(f"Retrieved document: {json.dumps(metadata, indent=2)}")
        context = "\n\n".join(formatted_context) if formatted_context else "No relevant information found."
        return context, sources
    except Exception as e:
        logger.error(f"Error formatting context: {e}")
        return "Error formatting context.", []

# Retrieve documents for multiple queries and combine results
def retrieve_documents(queries: list) -> list:
    """
    Retrieve documents from the FAISS vector store for a list of queries and combine unique results.
    
    Args:
        queries (list): List of query strings.
    
    Returns:
        list: List of unique Document objects.
    """
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        all_docs = []
        seen_ids = set()
        for query in queries:
            docs = retriever.invoke(query)
            for doc in docs:
                doc_id = f"{doc.metadata.get('title', 'unknown')}:{doc.metadata.get('page', '0')}"
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_docs.append(doc)
        return all_docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []

# Define answer generation chain
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
        You are an assistant for the New England First Amendment Coalition (NEFAC).
        Using the provided context from NEFAC's database (YouTube transcripts, summaries, documents), answer the user's question concisely and accurately.
        Reference video titles and timestamps (if available) to provide specific sources.
        If the context lacks sufficient information, say: "I don't have enough information from the NEFAC database to answer this fully."
        
        Context:
        {context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
answer_chain = answer_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1) | StrOutputParser()

# Main function to query the database and generate an answer (LEGACY)
def query_nefac_database(query: str, chat_history: list, session_id: str = "abc123") -> dict:
    """
    Rephrases the user's query, retrieves relevant documents using multiple query variations,
    and generates a single answer with source metadata.
    
    Args:
        query (str): The user's input query.
        chat_history (list): List of previous messages in the conversation.
        session_id (str): Session ID for chat history management.
    
    Returns:
        dict: Dictionary containing the answer and list of sources.
    """
    try:
        # Step 1: Rephrase the query
        rephrased_query = rephrase_query(query, chat_history, session_id)
        
        # Step 2: Generate multiple query variations
        query_variations = multi_query_chain.invoke({"query": rephrased_query})
        queries = [rephrased_query] + query_variations
        
        # Log query variations for debugging
        logger.info(f"Query variations: {queries}")
        
        # Step 3: Retrieve documents for all queries
        documents = retrieve_documents(queries)
        
        # Log retrieved documents
        logger.info(f"Retrieved {len(documents)} unique documents")
        
        # Step 4: Format context and sources
        context, sources = format_context_and_sources(documents)
        
        # Step 5: Generate answer
        input_data = {
            "question": query,
            "chat_history": chat_history,
            "context": context
        }
        answer = answer_chain.invoke(input_data)
        
        logger.info(f"Generated answer: {answer}")
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error querying NEFAC database: {e}")
        return {
            "answer": "An error occurred while querying the NEFAC database.",
            "sources": []
        }

# Function to rephrase the query (unchanged)
def rephrase_query(query: str, chat_history: list, session_id: str = "abc123") -> str:
    """
    Rephrases the user's query into a standalone question for querying the NEFAC database.
    
    Args:
        query (str): The user's input query.
        chat_history (list): List of previous messages in the conversation.
        session_id (str): Session ID for chat history management.
    
    Returns:
        str: Rephrased standalone question.
    """
    try:
        # Get or create session history
        history = get_session_history(session_id)
        
        # Update history with current query (optional, for context)
        if chat_history:
            history.messages = chat_history
        
        # Invoke the rephrasing chain
        input_data = {"question": query, "chat_history": history.messages}
        rephrased_question = contextualize_q_chain.invoke(input_data)
        
        logger.info(f"Original query: {query}")
        logger.info(f"Rephrased query: {rephrased_question}")
        
        return rephrased_question
    except Exception as e:
        logger.error(f"Error rephrasing query: {e}")
        # Fallback to original query if rephrasing fails
        return query

# Async generator function for streaming responses (UPDATED TO USE NEW IMPLEMENTATION)
async def middleware_qa(query: str, convoHistory: str = ""):
    """
    Async generator function that queries the NEFAC database and yields response chunks.
    This function serves as middleware between the main API and the query processing logic.
    NOW USES THE NEW CLEAN IMPLEMENTATION.
    
    Args:
        query (str): The user's input query.
        convoHistory (str): Conversation history as a string.
    
    Yields:
        str: JSON-encoded response chunks in the format expected by the frontend.
    """
    try:
        # Convert convoHistory string to list format expected by query_nefac_database_new
        chat_history = []
        if convoHistory:
            try:
                chat_history = json.loads(convoHistory)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse convoHistory: {convoHistory}")
                chat_history = []
        
        # Query the database using NEW implementation
        result = query_nefac_database_new(query, chat_history)
        
        # Also get the chunks for content display
        vector_queries = generate_vector_queries(query, chat_history)
        all_chunks = retrieve_chunks_from_queries(vector_queries, k_per_query=5)
        
        # Create a mapping of chunk info to chunk content for better matching
        chunk_map = {}
        for chunk in all_chunks:
            title = chunk.metadata.get('title', 'unknown')
            timestamp = chunk.metadata.get('page', 0)
            chunk_key = f"{title}:{timestamp}"
            chunk_map[chunk_key] = {
                'content': chunk.page_content,
                'metadata': chunk.metadata
            }
        
        # Transform the sources to match frontend expectations with simplified structure
        context_data = []
        for source in result.get("sources", []):
            title = source.get('title', 'Unknown')
            timestamp_seconds = source.get('timestamp_seconds')
            
            # Find the corresponding chunk content using better matching
            chunk_content = ""
            chunk_key = f"{title}:{timestamp_seconds}"
            if chunk_key in chunk_map:
                chunk_content = chunk_map[chunk_key]['content']
            else:
                # Fallback: look for any chunk with matching title
                for key, chunk_data in chunk_map.items():
                    if key.startswith(f"{title}:"):
                        chunk_content = chunk_data['content']
                        break
            
            # Ensure timestamp_seconds is properly passed through
            context_data.append({
                "title": title,
                "link": source.get("link", ""),
                "type": source.get("type", "unknown"),
                "timestamp_seconds": timestamp_seconds,
                "summary": source.get("summary", ""),
                "content": chunk_content
            })
            
            # Log for debugging
            logger.info(f"Source added: {title} at {timestamp_seconds}s, link: {source.get('link', '')}")
        
        # First yield the context/sources in SSE format
        if context_data:
            context_chunk = {
                "context": context_data,
                "order": 1
            }
            yield f"data: {json.dumps(context_chunk)}\n\n"
        
        # Then yield the main message in SSE format
        message_chunk = {
            "message": result.get("answer", "No response available."),
            "order": 2
        }
        yield f"data: {json.dumps(message_chunk)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in middleware_qa: {e}")
        error_chunk = {
            "message": "An error occurred while processing your query.",
            "order": 1
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"