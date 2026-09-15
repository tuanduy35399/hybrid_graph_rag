from dataclasses import dataclass
from enum import Enum

from src.rag.normalization.types import NormalizationResult


class BranchType(str, Enum):
    IMAGE_BRANCH = "image_branch"
    TEXT_ONLY_BRANCH = "text_only_branch"


@dataclass(frozen=True)
class PreparedBranchInput:
    branch: BranchType
    normalization: NormalizationResult
