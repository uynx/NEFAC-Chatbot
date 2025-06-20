from langchain_community.document_loaders import YoutubeLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def youtubeLoader(url, title_to_chunks, url_to_title):
    """Load YouTube video transcript using langchain YoutubeLoader"""
    try:
        # Use langchain's YoutubeLoader to get transcript and metadata
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        docs = loader.load()
        
        if not docs:
            logger.warning(f"No transcript found for URL: {url}")
            return set()
        
        # Get the document (transcript)
        doc = docs[0]
        
        # Extract title from metadata, fallback to a default if not available
        title = doc.metadata.get('title', f'YouTube Video {url}')
        
        # Check if we already have this title
        if title in title_to_chunks:
            logger.info(f"Video already processed: {title}")
            return set()
        
        # Set metadata to match what chain.py expects
        doc.metadata['title'] = title
        doc.metadata['type'] = 'youtube'
        doc.metadata['source'] = url
        
        # Store in our mappings
        title_to_chunks[title] = docs
        url_to_title[url] = title
        
        logger.info(f"Successfully loaded YouTube video: {title}")
        return {title}
        
    except Exception as e:
        logger.error(f"Error loading YouTube video {url}: {str(e)}")
        return set()