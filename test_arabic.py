from pypdf import PdfReader

reader = PdfReader("../pdfs/solar_energy_arabic.pdf")

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    print(f"\n--- Page {page_number} ---")
    print(text)