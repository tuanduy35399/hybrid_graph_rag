from llama_index.core.extractors import (
    TitleExtractor,
    KeywordExtractor,
    SummaryExtractor,
)
from llama_index.core.ingestion import IngestionPipeline

#Buoc nay trich xuat metadata tu dong bang llama index thay vi hardcore viet metadata cung
class MetadataExtractor:
    def __init__(self):
        self.pipeline = IngestionPipeline(
            transformations=[
                TitleExtractor(nodes=3), 
                #TitleExtractor khong lay tieu de, ma dung 3 node dau de suy luan tieu de chung
                #de sau nay coi cho nay co the dieu chinh tang giam so node 
                KeywordExtractor(keywords=3), #sinh ra 5 tu khoa
                SummaryExtractor(summaries=["self"]), 
                #tom tat nut truoc, ban than no, nut sau no
            ]
        )

    def extract(self, nodes):
        return self.pipeline.run(nodes=nodes)