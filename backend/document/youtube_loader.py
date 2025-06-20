import yt_dlp
import requests
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  
from load_env import load_env
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_env()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=1024,
    api_key=os.getenv("OPENAI_API_KEY") 
)

def clean_text(text):
    """Clean transcript text using LLM"""
    prompt_template = PromptTemplate.from_template(
        """
        Clean this YouTube transcript for NEFAC. Fix grammar, remove filler words, remove [Music]/[Applause], fix "kneefact" to "NEFAC", and use context for homophones (metal->medal, there/their/they're):

        Raw: "{input_text}"
        Cleaned:
        """
    )

    try:
        cleaned_text = llm.invoke(prompt_template.format(input_text=text)).content.strip()
        return cleaned_text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text

def youtubeLoader(url, title_to_chunks, url_to_title):
    """Load YouTube video using yt-dlp for metadata and youtube-transcript-api for time-based chunking"""
    try:
        # Use yt-dlp to get video metadata (title, etc.)
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'vtt',
            'subtitleslangs': ['en'],
            'skip_download': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            title = info.get('title', f'YouTube Video {url}')
            
            # Check if already processed
            if title in title_to_chunks:
                logger.info(f"Video already processed: {title}")
                return set()
            
            # Get transcript with timing data using YouTube Transcript API
            from youtube_transcript_api import YouTubeTranscriptApi
            from urllib.parse import parse_qs, urlparse
            
            # Extract video ID from URL
            parsed_url = urlparse(url)
            if parsed_url.hostname in ['youtu.be']:
                video_id = parsed_url.path[1:]
            elif parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
                video_id = parse_qs(parsed_url.query)['v'][0]
            else:
                logger.warning(f"Invalid YouTube URL: {url}")
                return set()
            
            # Get transcript with timing data from YouTube Transcript API
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # Prefer manual subtitles over auto-generated (higher quality)
                transcript_data = None
                for transcript in transcript_list:
                    if transcript.language_code.startswith('en'):
                        if not transcript.is_generated:  # Manual transcripts first
                            transcript_data = transcript.fetch()
                            logger.info(f"Using manual subtitles for: {title}")
                            break
                
                # If no manual English found, try auto-generated
                if not transcript_data:
                    for transcript in transcript_list:
                        if transcript.language_code.startswith('en') and transcript.is_generated:
                            transcript_data = transcript.fetch()
                            logger.info(f"Using automatic captions for: {title}")
                            break
                
                if not transcript_data:
                    logger.warning(f"No English transcript found for: {title}")
                    return set()
                
            except Exception as e:
                logger.error(f"Failed to get transcript for {url}: {e}")
                return set()
            
            logger.info(f"Retrieved {len(transcript_data)} transcript entries")
            
            # Create time-based chunks using actual transcript timestamps (60 seconds each)
            chunks = []
            current_chunk = []
            current_start = 0
            chunk_duration = 60  # 60 seconds per chunk
            
            for entry in transcript_data:
                start_time = entry.start
                text = entry.text
                
                # When we reach 60 seconds, finalize the current chunk
                if start_time - current_start >= chunk_duration and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    
                    # Log raw text and clean it with LLM
                    logger.info(f"Chunk {len(chunks)+1} raw (first 15 words): {' '.join(chunk_text.split()[:15])}")
                    cleaned_chunk_text = clean_text(chunk_text)
                    logger.info(f"Chunk {len(chunks)+1} cleaned (first 15 words): {' '.join(cleaned_chunk_text.split()[:15])}")
                    
                    doc = Document(
                        page_content=cleaned_chunk_text,
                        metadata={
                            'title': title,
                            'type': 'youtube',
                            'source': url,
                            'page': int(current_start),
                            'start_seconds': int(current_start),
                            'end_seconds': int(start_time)
                        }
                    )
                    chunks.append(doc)
                    current_chunk = []
                    current_start = start_time
                
                current_chunk.append(text)
            
            # Add the last chunk if there's content
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                logger.info(f"Final chunk raw (first 15 words): {' '.join(chunk_text.split()[:15])}")
                cleaned_chunk_text = clean_text(chunk_text)
                logger.info(f"Final chunk cleaned (first 15 words): {' '.join(cleaned_chunk_text.split()[:15])}")
                
                doc = Document(
                    page_content=cleaned_chunk_text,
                    metadata={
                        'title': title,
                        'type': 'youtube', 
                        'source': url,
                        'page': int(current_start),
                        'start_seconds': int(current_start),
                        'end_seconds': int(current_start + 60)  # Estimate for last chunk
                    }
                )
                chunks.append(doc)
            
            logger.info(f"Created {len(chunks)} time-based chunks from transcript")
            
            # Store in mappings
            title_to_chunks[title] = chunks
            url_to_title[url] = title
            
            logger.info(f"Successfully processed YouTube video: {title} ({len(chunks)} chunks)")
            return {title}
    
    except Exception as e:
        logger.error(f"Error loading YouTube video {url}: {e}")
        return set()