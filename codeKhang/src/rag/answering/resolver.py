from pathlib import Path

from src.rag.retrieval.manifest_indexer import load_manifest_records


class ManifestChunkResolver:
    def __init__(self, records_by_id):
        self.records_by_id = dict(records_by_id)

    def resolve(self, chunk_ids: tuple[str, ...]):
        return tuple(
            self.records_by_id[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self.records_by_id
        )


def build_manifest_chunk_resolver(manifest_path: Path) -> ManifestChunkResolver:
    records = load_manifest_records(manifest_path)
    records_by_id = {}
    for record in records:
        if record.chunk_id in records_by_id:
            raise ValueError(f"Duplicate chunk_id '{record.chunk_id}' found in manifest records.")
        records_by_id[record.chunk_id] = record
    return ManifestChunkResolver(records_by_id)
