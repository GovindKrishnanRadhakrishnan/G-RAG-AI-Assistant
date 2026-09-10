import arxiv
from scholarly import scholarly
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
from langchain.tools import BaseTool
from pydantic import Field
import json

class PaperSearchTool(BaseTool):
    name = "paper_search"
    description = "Search for academic papers across arXiv and Google Scholar"
    max_results: int = Field(default=10, description="Maximum number of results to return")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="List of sources found")

    def _run(self, query: str) -> str:
        """Search for papers using the query."""
        try:
            # Search arXiv
            arxiv_search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            arxiv_results = []
            for result in arxiv_search.results():
                arxiv_results.append({
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "summary": result.summary,
                    "pdf_url": result.pdf_url,
                    "published": str(result.published),
                    "source": "arXiv"
                })
            
            # Search Google Scholar
            scholar_results = []
            search_query = scholarly.search_pubs(query)
            for i, result in enumerate(search_query):
                if i >= self.max_results:
                    break
                scholar_results.append({
                    "title": result.bib.get('title', ''),
                    "authors": result.bib.get('author', []),
                    "abstract": result.bib.get('abstract', ''),
                    "url": result.bib.get('url', ''),
                    "year": result.bib.get('year', ''),
                    "source": "Google Scholar"
                })
            
            # Combine results
            self.sources = arxiv_results + scholar_results
            
            return json.dumps({
                "status": "success",
                "results": self.sources
            }, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e)
            }, indent=2)

    def _arun(self, query: str) -> str:
        """Async version of _run."""
        return self._run(query)

    def get_sources(self) -> List[Dict]:
        """Get the list of sources found in the last search."""
        return self.sources