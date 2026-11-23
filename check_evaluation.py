from vectoredb import get_all_chunks

chunks = get_all_chunks()

print("Available sources:\n")

for source in sorted(set(chunk["source"] for chunk in chunks)):
    print(repr(source))