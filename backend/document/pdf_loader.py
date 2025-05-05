import glob
from langchain_community.document_loaders import PyPDFLoader
from document.summary import generate_summary, generate_tags
import os

def pdfLoader(pdf_path, title_to_chunks):
    doc_title = os.path.basename(pdf_path)[:-4].strip().replace('_', ' ').replace('  ', ' ')
    if doc_title in title_to_chunks:
        return set()  # Skip if already processed
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    # summary = generate_summary(pages)
    # tags = generate_tags(summary)
    for page in pages:
        page.metadata['title'] = doc_title
        page.metadata['type'] = 'pdf'
        title_to_chunks[doc_title] = pages
        # page.metadata['summary'] = summary
        # page.metadata['audience'] = tags.get('audience', [])
        # page.metadata['nefac_category'] = tags.get('nefac_category', [])
        # page.metadata['resource_type'] = tags.get('resource_type', [])
    return {doc_title}