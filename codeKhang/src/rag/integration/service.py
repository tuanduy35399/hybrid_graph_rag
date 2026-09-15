from src.rag.integration.types import BranchType, PreparedBranchInput
from src.rag.normalization.service import LLMNormalizer


class SharedBranchPreprocessor:
    def __init__(self, normalizer: LLMNormalizer):
        self.normalizer = normalizer

    def prepare(self, raw_text: str, has_image: bool) -> PreparedBranchInput:
        normalization = self.normalizer.normalize(raw_text)
        branch = BranchType.IMAGE_BRANCH if has_image else BranchType.TEXT_ONLY_BRANCH
        return PreparedBranchInput(branch=branch, normalization=normalization)
