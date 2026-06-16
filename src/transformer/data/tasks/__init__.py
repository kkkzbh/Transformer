"""Task plugins."""

from transformer.data.tasks.base import Seq2SeqTask
from transformer.data.tasks.reverse import ReverseTask

__all__ = ["ReverseTask", "Seq2SeqTask"]
