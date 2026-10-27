class HybridSearch:

    def __init__(self):
        pass

    def merge(self, vector_results, graph_results):
        merged = []
        seen = set()

        print("Bat dau ket hop 2 search")
        for node in (vector_results or []):
            node_obj = getattr(node, "node", node)
            node_id = getattr(node_obj, "node_id", getattr(node_obj, "id_", str(node_obj)))

            if node_id not in seen:
                merged.append(node)
                seen.add(node_id)

        for node in (graph_results or []):
            node_obj = getattr(node, "node", node)
            node_id = getattr(node_obj, "node_id", getattr(node_obj, "id_", str(node_obj)))

            if node_id not in seen:
                merged.append(node)
                seen.add(node_id)

        return merged