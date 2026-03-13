# rag.py

import json
import os

# Simple in-memory store: { clinic_id: [chunks] }
clinic_data_store = {}


def load_clinic_data(clinic_id: str, file_path: str):
    """Load a clinic's text file and split into chunks."""
    with open(file_path, 'r') as f:
        text = f.read()

    # Split by double newline — each paragraph is one chunk
    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]

    # Store in memory
    clinic_data_store[clinic_id] = chunks

    print(f"Loaded {len(chunks)} chunks for clinic: {clinic_id}")
    return len(chunks)


def search_clinic_data(clinic_id: str, query: str, n_results: int = 3):
    """
    Search clinic data for chunks relevant to the query.
    Simple keyword matching — finds chunks containing query words.
    """
    if clinic_id not in clinic_data_store:
        return []

    chunks = clinic_data_store[clinic_id]
    query_words = query.lower().split()

    # Score each chunk by how many query words it contains
    scored = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for word in query_words if word in chunk_lower)
        if score > 0:
            scored.append((score, chunk))

    # Sort by score, return top results
    scored.sort(reverse=True)
    top_chunks = [chunk for score, chunk in scored[:n_results]]

    # If nothing matched, return first 2 chunks as fallback
    if not top_chunks:
        return chunks[:2]

    return top_chunks
