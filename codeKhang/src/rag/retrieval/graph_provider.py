from pathlib import Path

from src.rag.retrieval.manifest_indexer import load_manifest_records


class ManifestGraphChunkProvider:
    def __init__(self, records):
        self._records = tuple(records)

    def get_chunks(self, retrieval_request):
        return self._records


def build_manifest_graph_chunk_provider(
    manifest_path: Path = Path("manifest.jsonl"),
):
    records = load_manifest_records(manifest_path)
    return ManifestGraphChunkProvider(records)
