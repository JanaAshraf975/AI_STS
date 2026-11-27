
from vectoredb import get_all_chunks, index_chunks, count

OLD_COLLECTION = "docs"
NEW_COLLECTION = "docs_ip"

print("Reading existing chunks...")

# Get all chunks from the old collection
chunks = get_all_chunks(
    collection_name=OLD_COLLECTION
)

print(f"Found {len(chunks)} chunks in {OLD_COLLECTION}")

if not chunks:
    raise ValueError("No chunks found in the old collection!")

# Re-index chunks into the new collection
# reset=True deletes only docs_ip before indexing
index_chunks(
    chunks=chunks,
    reset=True,
    batch=256,
    collection_name=NEW_COLLECTION
)

# Verify indexing
new_count = count() if NEW_COLLECTION == "docs" else None

from vectoredb import get_collection

new_collection = get_collection(NEW_COLLECTION)

print("\n========== RE-INDEX REPORT ==========")
print(f"Old collection: {OLD_COLLECTION}")
print(f"Old chunks: {len(chunks)}")
print(f"New collection: {NEW_COLLECTION}")
print(f"New chunks: {new_collection.count()}")

assert new_collection.count() == len(chunks), (
    "Re-indexing verification failed!"
)

print("Re-indexing completed successfully!")
