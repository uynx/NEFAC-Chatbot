#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))

from document.youtube_loader import (
    extract_video_id, 
    check_video_availability, 
    get_transcript_direct,
    get_youtube_title,
    youtubeLoader
)
from youtube_transcript_api import YouTubeTranscriptApi

def comprehensive_transcript_test(url):
    """More thorough transcript detection"""
    video_id = extract_video_id(url)
    if not video_id:
        return None, "Invalid video ID"
    
    try:
        # Get list of all available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        print(f"Available transcripts for video {video_id}:")
        available_transcripts = []
        
        for transcript in transcript_list:
            lang_info = f"Language: {transcript.language} ({transcript.language_code})"
            if transcript.is_generated:
                lang_info += " [Auto-generated]"
            else:
                lang_info += " [Manual]"
            print(f"  - {lang_info}")
            available_transcripts.append((transcript.language_code, transcript.language, transcript.is_generated))
        
        if not available_transcripts:
            return None, "No transcripts found"
        
        # Try to get the first available transcript
        for lang_code, lang_name, is_generated in available_transcripts:
            try:
                transcript = transcript_list.find_transcript([lang_code])
                transcript_data = transcript.fetch()
                return transcript_data, f"Found transcript in {lang_name} ({lang_code}) - {'Auto-generated' if is_generated else 'Manual'}"
            except Exception as e:
                print(f"  Failed to fetch {lang_name}: {e}")
                continue
        
        return None, "All transcript fetches failed"
        
    except Exception as e:
        return None, f"Transcript list error: {str(e)}"

def test_single_video(url):
    print(f"Testing YouTube URL: {url}")
    print(f"{'='*80}")
    
    # Test 1: Extract video ID
    video_id = extract_video_id(url)
    print(f"Video ID: {video_id}")
    
    # Test 2: Check availability
    is_available, availability_msg = check_video_availability(url)
    print(f"Available: {is_available} - {availability_msg}")
    
    # Test 3: Get title
    title = get_youtube_title(url)
    print(f"Title: {title}")
    
    # Test 4: Comprehensive transcript test
    if is_available:
        print(f"\n{'='*40}")
        print("COMPREHENSIVE TRANSCRIPT TEST")
        print(f"{'='*40}")
        
        transcript_data, transcript_msg = comprehensive_transcript_test(url)
        print(f"Result: {transcript_msg}")
        
        if transcript_data:
            print(f"Transcript entries: {len(transcript_data)}")
            if len(transcript_data) > 0:
                print(f"First entry: {transcript_data[0]}")
                print(f"Sample text: {transcript_data[0]['text'][:200]}...")
                print(f"Last entry: {transcript_data[-1]}")
    
    # Test 5: Try our original direct method for comparison
    print(f"\n{'='*40}")
    print("ORIGINAL DIRECT TRANSCRIPT TEST")
    print(f"{'='*40}")
    
    if is_available:
        transcript_data_orig, transcript_msg_orig = get_transcript_direct(url)
        print(f"Original method result: {transcript_msg_orig}")
    
    # Test 6: Try full YouTube loader
    print(f"\n{'='*40}")
    print("FULL YOUTUBE LOADER TEST")
    print(f"{'='*40}")
    
    title_to_chunks = {}
    url_to_title = {}
    
    try:
        result = youtubeLoader(url, title_to_chunks, url_to_title)
        print(f"Loader result: {result}")
        
        if result:
            for title, docs in title_to_chunks.items():
                print(f"Loaded: {title}")
                print(f"Documents: {len(docs)}")
                if docs and len(docs) > 0:
                    print(f"First doc content preview: {docs[0].page_content[:200]}...")
                    print(f"First doc metadata: {docs[0].metadata}")
    except Exception as e:
        print(f"YouTube loader error: {e}")
    
    return is_available, transcript_data is not None if is_available else False

if __name__ == "__main__":
    # Test the new URL
    test_url = "https://www.youtube.com/watch?v=DxFlbg5WYmc&pp=ygUHc3N1bmRlZQ%3D%3D"
    
    available, has_transcript = test_single_video(test_url)
    
    print(f"\n{'='*80}")
    print("FINAL RESULT")
    print(f"{'='*80}")
    
    status = "✅ SUCCESS" if available and has_transcript else "❌ NO TRANSCRIPT" if available else "❌ NOT AVAILABLE"
    video_id = extract_video_id(test_url)
    print(f"{status} - Video ID: {video_id}")
    print(f"Available: {available}")
    print(f"Has Transcript: {has_transcript}") 