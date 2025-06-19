#!/usr/bin/env python3
from document.youtube_loader import youtubeLoader

# Test with empty dictionaries
title_to_chunks = {}
url_to_title = {}

print('Testing first video...')
result1 = youtubeLoader('https://www.youtube.com/watch?v=YwnzNw80-84', title_to_chunks, url_to_title)
print(f'Result 1: {result1}')

print('\nTesting second video...')
result2 = youtubeLoader('https://www.youtube.com/watch?v=LalaGFCElXA', title_to_chunks, url_to_title)
print(f'Result 2: {result2}')

print(f'\nTotal titles loaded: {len(title_to_chunks)}')
for title, docs in title_to_chunks.items():
    print(f'  {title}: {len(docs)} documents')
    print(f'    First doc type: {docs[0].metadata.get("type", "unknown")}')
    print(f'    Has transcript: {docs[0].metadata.get("transcript_available", "unknown")}') 