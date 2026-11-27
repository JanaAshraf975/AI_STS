from vectoredb import get_all_chunks

chunks = get_all_chunks()

target_ids = {
    "nova_bikes_handbook_p3_c0",
    "nova_bikes_handbook_p3_c1",
    "nova_bikes_handbook_p3_c2",
    "nova_bikes_handbook_p50_c0",
    "nova_bikes_handbook_p50_c1",
}

for chunk in chunks:
    if chunk["id"] in target_ids:
        print("\n" + "=" * 80)
        print("ID:", chunk["id"])
        print("TEXT:", chunk["text"])