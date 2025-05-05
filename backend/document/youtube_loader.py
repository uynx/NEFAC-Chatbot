import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from document.summary import generate_summary, generate_tags
import yt_dlp

def get_youtube_title(url):
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
        print(f"Error fetching title for {url}: {str(e)}")
        return "Title not found"

def youtubeLoader(url, title_to_chunks, url_to_title):
    # if url in url_to_title:

    if url in url_to_title:
        title=url_to_title[url]
        if title in title_to_chunks:
            return set()
    title=get_youtube_title(url)
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False,
        transcript_format=TranscriptFormat.CHUNKS,
        chunk_size_seconds=60,
    )
    loaded_clips = loader.load()
    # summary = generate_summary(loaded_clips[:-1]) # cut off the last minute
    # tags = generate_tags(summary,youtube=True)
    for clip in loaded_clips:
        clip.metadata["title"] = title
        clip.metadata["page"] = clip.metadata["start_seconds"]
        clip.metadata['type'] = 'youtube'
        # clip.metadata['audience'] = tags.get('audience', [])
        # clip.metadata['nefac_category'] = tags.get('nefac_category', [])
        # clip.metadata['resource_type'] = tags.get('resource_type', [])
        # clip.metadata['summary'] = summary
    title_to_chunks[title] = loaded_clips
    return {title}