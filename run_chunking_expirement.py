
from pdf_utils import chunk_pdf
from vectoredb import index_chunks

PDF_PATH = "../pdfs/solar_energy_arabic.pdf"

experiments = [
    {
        "name": "A",
        "chunk_size": 800,
        "overlap": 150,
        "collection": "solar_800_150"
    },
    {
        "name": "B",
        "chunk_size": 500,
        "overlap": 100,
        "collection": "solar_500_100"
    },
    {
        "name": "C",
        "chunk_size": 1000,
        "overlap": 200,
        "collection": "solar_1000_200"
    }
]

for exp in experiments:
    print(f"\nStarting Experiment {exp['name']}")

    chunks = chunk_pdf(
        PDF_PATH,
        chunk_size=exp["chunk_size"],
        overlap=exp["overlap"]
    )

    index_chunks(
        chunks,
        reset=True,
        collection_name=exp["collection"]
    )

    print(f"Collection: {exp['collection']}")
    print(f"Number of chunks: {len(chunks)}")

print("\nAll experiments indexed successfully!")
