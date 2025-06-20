import yt_dlp
import requests
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from load_env import load_env
import os
import logging
import json
from urllib.parse import parse_qs, urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_env()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens=1024, api_key=os.getenv("OPENAI_API_KEY"))

def clean_text(text):
    """Clean transcript text using LLM"""
    prompt = """You are a professional transcript editor specializing in cleaning auto-generated YouTube transcripts for NEFAC (New England First Amendment Coalition). Your task is to:
    1. Correct grammar, punctuation, and spelling errors.
    2. Remove filler words (e.g., "um," "uh," "like") and redundant phrases.
    3. Remove YouTube-specific artifacts (e.g., "[Music]," "[Applause]").
    4. Standardize proper names to their most likely correct form.
    5. Ensure the text is clear, concise, and preserves the original meaning.
    6. Return only the cleaned text, without additional explanations.
    7. Fix all spellings of NEFAC (e.g. kneefact -> NEFAC)
    8. Use context to correct spelling mistakes and homophones (e.g., "metal" should be "medal" when discussing Olympic medals, "there" vs "their" vs "they're" based on context)

    Examples:
    Raw: "Um, so like, we're gonna talk about, uh, AI today and stuff."
    Cleaned: We're going to talk about AI today.

    Raw: "The, the thing is is that, uh, machine learning is, like, super cool."
    Cleaned: The thing is that machine learning is very cool.

    Raw: "Okay, let's see.. data science is, um, important. For for example, it helps with, uh, predictions."
    Cleaned: Data science is important. For example, it helps with predictions.

    Raw: "Next, uh, [Music] we discuss open meetings with John Maran or Marian."
    Cleaned: Next, we discuss open meetings with John Marian.

    Raw: "kneefact has been working on a new project."
    Cleaned: NEFAC has been working on a new project.

    Raw: "The athlete won a gold metal in the Olympics there performance was amazing."
    Cleaned: The athlete won a gold medal in the Olympics their performance was amazing.

    Now, clean the following transcript text:
    Raw: "{input_text}"
    Cleaned:"""
    
    try:
        return llm.invoke(PromptTemplate.from_template(prompt).format(input_text=text)).content.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text

def get_youtube_title(url):
    """Get YouTube title using yt-dlp"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Title not found')
            return title
    except Exception as e:
        logger.error(f"Error fetching title for {url}: {str(e)}")
        return "Title not found"

def youtubeLoader(url, title_to_chunks, url_to_title):
    """Load YouTube video using Langchain YoutubeLoader with 60-second chunks"""
    
    try:
        # Check if already processed
        if url in url_to_title:
            title = url_to_title[url]
            if title in title_to_chunks:
                logger.info(f"Already processed: {title}")
                return set()
        else:
            title = get_youtube_title(url)
            url_to_title[url] = title
            logger.info(f"Processing: {title}")
        
        # Use Langchain YoutubeLoader for transcript extraction
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,
            transcript_format=TranscriptFormat.CHUNKS,
            chunk_size_seconds=60,
        )
        
        loaded_clips = loader.load()
        
        if not loaded_clips:
            logger.warning(f"No transcript found for: {title}")
            return set()
        
        logger.info(f"Found {len(loaded_clips)} chunks for: {title}")
        
        # Clean and process each chunk
        cleaned_chunks = []
        for i, clip in enumerate(loaded_clips):
            raw_text = clip.page_content
            cleaned_text = clean_text(raw_text)
            
            logger.info(f"Chunk {i+1} - Raw: {' '.join(raw_text.split()[:15])}")
            logger.info(f"Chunk {i+1} - Cleaned: {' '.join(cleaned_text.split()[:15])}")
            
            # Update metadata
            clip.page_content = cleaned_text
            clip.metadata["title"] = title
            clip.metadata["page"] = clip.metadata.get("start_seconds", i * 60)
            clip.metadata["type"] = "youtube"
            clip.metadata["source"] = url
            
            # Ensure start_seconds and end_seconds are in metadata
            if "start_seconds" not in clip.metadata:
                clip.metadata["start_seconds"] = i * 60
            if "end_seconds" not in clip.metadata:
                clip.metadata["end_seconds"] = (i + 1) * 60
                
            cleaned_chunks.append(clip)
        
        # Save results
        title_to_chunks[title] = cleaned_chunks
        url_to_title[url] = title
        
        logger.info(f"Successfully processed {len(cleaned_chunks)} chunks for: {title}")
        return {title}
        
    except Exception as e:
        logger.error(f"Error processing {url}: {e}")
        return set()