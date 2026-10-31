from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pages_from_pdf(pdf_path):
    """Returns a list of {"page": int, "text": str} for each page."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_pdf(pdf_path, chunk_size=800, overlap=150):
    """Chunks a PDF page by page and keeps source metadata for each chunk."""
    pdf_path = Path(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "؟ ", "! ", "، ", " ", ""],
    )

    chunks = []
    for page in extract_pages_from_pdf(pdf_path):
        for j, piece in enumerate(splitter.split_text(page["text"])):
            piece = piece.strip()
            if len(piece) > 30:
                chunks.append({
                    "id": f"{pdf_path.stem}_p{page['page']}_c{j}",
                    "text": piece,
                    "source": pdf_path.name,
                    "page": page["page"],
                })
    return chunks


def chunk_folder(folder):
    """Chunks every PDF in a folder."""
    all_chunks = []
    for pdf in sorted(Path(folder).glob("*.pdf")):
        all_chunks.extend(chunk_pdf(pdf))
    return all_chunks
