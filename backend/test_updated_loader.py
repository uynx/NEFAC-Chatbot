#!/usr/bin/env python3
"""
Test the updated YouTube loader with yt-dlp fallback
"""

from document.youtube_loader import youtubeLoader

def test_updated_loader():
    """Test the updated YouTube loader"""
    
    test_url = "https://www.youtube.com/watch?v=DxFlbg5WYmc&pp=ygUHc3N1bmRlZQ%3D%3D"
    
    # Initialize storage dicts
    title_to_chunks = {}
    url_to_title = {}
    
    print("🧪 Testing updated YouTube loader with yt-dlp fallback...")
    print(f"📹 Test URL: {test_url}")
    print("="*70)
    
    # Test the loader
    try:
        new_titles = youtubeLoader(test_url, title_to_chunks, url_to_title)
        
        if new_titles:
            title = list(new_titles)[0]
            chunks = title_to_chunks.get(title, [])
            
            print(f"\n✅ SUCCESS!")
            print(f"📝 Title: {title}")
            print(f"📄 Chunks created: {len(chunks)}")
            
            if chunks:
                first_chunk = chunks[0]
                print(f"📊 First chunk metadata: {first_chunk.metadata}")
                print(f"🗣️ First chunk content (preview): {first_chunk.page_content[:200]}...")
                
                # Check if transcript was available
                if hasattr(first_chunk.metadata, 'transcript_available') and not first_chunk.metadata.transcript_available:
                    print("⚠️ Note: This is a metadata-only document (no transcript)")
                else:
                    print("✅ Full transcript successfully extracted!")
            
            return True
        else:
            print("❌ FAILED: No titles returned")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_updated_loader()
    
    print("\n" + "="*70)
    if success:
        print("🎉 Updated YouTube loader is working!")
        print("💡 yt-dlp fallback successfully integrated")
    else:
        print("😞 Updated loader still has issues")
        print("💡 Further debugging needed") 