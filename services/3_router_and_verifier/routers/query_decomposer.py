from typing import List

class QueryDecomposer:
    """
    Decomposes complex or comparative questions into independent sub-queries.
    """
    def decompose(self, query: str) -> List[str]:
        # Split on comparison indicators
        for sep in [" vs ", " versus ", " مقارنة بين ", " مقارنة مع ", " الفرق بين "]:
            if sep in query:
                parts = query.split(sep)
                return [p.strip() for p in parts if p.strip()]
        return [query]
