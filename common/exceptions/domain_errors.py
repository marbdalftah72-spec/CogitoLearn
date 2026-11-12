"""
Custom domain exceptions for CogitoLearn system.
"""

class CogitoLearnException(Exception):
    """Base exception for all domain errors."""
    pass

class IngestionError(CogitoLearnException):
    """Raised when parsing documents or transcribing audio fails."""
    pass

class EmbeddingError(CogitoLearnException):
    """Raised when vector embedding generation fails."""
    pass

class KnowledgeGraphError(CogitoLearnException):
    """Raised during concept extraction or prerequisite graph traversal."""
    pass

class RetrievalAbstentionError(CogitoLearnException):
    """Raised when evidence coverage is insufficient for calibrated answering."""
    pass

class VerificationMismatchError(CogitoLearnException):
    """Raised when NLI or algebraic verification fails against retrieved spans."""
    pass

class PedagogyStateError(CogitoLearnException):
    """Raised during invalid BKT updates or FSRS scheduling errors."""
    pass
