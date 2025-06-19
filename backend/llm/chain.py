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
            "answer": "I'm sorry, but NEFAC doesn't have any information about that topic in our current database.",
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

        Your job is to answer questions using information from NEFAC's database.

        HOW TO RESPOND:
        1. If you find relevant information in the sources below, use it to answer the question. Reference specific sources (e.g., "According to the Data Cleaning 101 video").
        
        2. If sources mention the topic indirectly, you can explain the concept based on how NEFAC discusses it. Start with "Based on NEFAC's materials..." 
        
        3. If the topic is unrelated to NEFAC's work (like pizza, sports, etc.), say: "That topic isn't related to NEFAC's focus on First Amendment rights and government transparency. I can help with journalism, public records, FOI requests, and related topics."
        
        4. If the topic is relevant but not found, say: "I'm sorry, but NEFAC doesn't have information about [topic] in our current database."

        CRITICAL: End every response with a new line containing "SOURCES_USED:" followed by the source numbers you used (e.g., "SOURCES_USED: 1, 3, 5") or "SOURCES_USED: none" if you found no relevant information.

        Available sources:
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
        logger.info(f"Raw LLM response: '{full_response}'")
        
        # Parse the response to separate answer and used sources
        if "SOURCES_USED:" in full_response:
            parts = full_response.split("SOURCES_USED:")
            answer = parts[0].strip()
            sources_used_text = parts[1].strip() if len(parts) > 1 else "none"
        else:
            answer = full_response.strip()
            sources_used_text = "none"
            logger.warning(f"LLM response missing SOURCES_USED, appending it. Response was: '{full_response}'")
        
        # Validate that we have a non-empty answer
        if not answer or len(answer.strip()) == 0:
            logger.warning("Generated empty answer, providing fallback response")
            answer = "I'm sorry, but NEFAC doesn't have any information about that topic in our current database."
            sources_used_text = "none"
        
        logger.info(f"Parsed answer: '{answer}'")
        logger.info(f"Sources used text: '{sources_used_text}'")
        
        # Determine which sources to include
        relevant_sources = []
        
        # Check if the answer indicates insufficient information
        insufficient_info_phrases = [
            "I'm sorry, but NEFAC doesn't have",
            "That topic isn't related to NEFAC",
            "don't have enough information",
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

async def middleware_qa(query: str, convoHistory: str = ""):
    try:
        chat_history = []
        if convoHistory:
            try:
                chat_history = json.loads(convoHistory)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse convoHistory: {convoHistory}")
                chat_history = []
        
        logger.info(f"Starting middleware_qa for query: {query}")
        result = query_nefac_database_new(query, chat_history)
        logger.info(f"Got result from query_nefac_database_new: {result}")
        
        # Build context data for sources
        context_data = []
        if result.get("sources"):
            # Get the chunks again to build context data
            vector_queries = generate_vector_queries(query, chat_history)
            all_chunks = retrieve_chunks_from_queries(vector_queries, k_per_query=5)
            chunk_map = {}
            for chunk in all_chunks:
                title = chunk.metadata.get('title', 'unknown')
                timestamp = chunk.metadata.get('page', 0)
                chunk_key = f"{title}:{timestamp}"
                chunk_map[chunk_key] = {
                    'content': chunk.page_content,
                    'metadata': chunk.metadata
                }
            
            for source in result.get("sources", []):
                title = source.get('title', 'Unknown')
                timestamp_seconds = source.get('timestamp_seconds')
                
                chunk_content = ""
                chunk_key = f"{title}:{timestamp_seconds}"
                if chunk_key in chunk_map:
                    chunk_content = chunk_map[chunk_key]['content']
                else:
                    for key, chunk_data in chunk_map.items():
                        if key.startswith(f"{title}:"):
                            chunk_content = chunk_data['content']
                            break
                
                context_data.append({
                    "title": title,
                    "link": source.get("link", ""),
                    "type": source.get("type", "unknown"),
                    "timestamp_seconds": timestamp_seconds,
                    "summary": source.get("summary", ""),
                    "content": chunk_content
                })
                
                logger.info(f"Source added: {title} at {timestamp_seconds}s, link: {source.get('link', '')}")
        
        # Send context data if we have sources
        if context_data:
            context_chunk = {
                "context": context_data,
                "order": 1
            }
            logger.info(f"Yielding context chunk: {context_chunk}")
            yield f"data: {json.dumps(context_chunk)}\n\n"
        
        # Always send the message
        answer = result.get("answer", "No response available.")
        message_chunk = {
            "message": answer,
            "order": 2
        }
        logger.info(f"Yielding message chunk: {message_chunk}")
        yield f"data: {json.dumps(message_chunk)}\n\n"
        
    except Exception as e:
        logger.error(f"Error in middleware_qa: {e}")
        error_chunk = {
            "message": "An error occurred while processing your query.",
            "order": 1
        }
        logger.info(f"Yielding error chunk: {error_chunk}")
        yield f"data: {json.dumps(error_chunk)}\n\n"