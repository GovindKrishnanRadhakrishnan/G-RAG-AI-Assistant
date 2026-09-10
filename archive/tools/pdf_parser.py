from PyPDF2 import PdfReader
from typing import Dict, Pattern
import re
import io
from langchain.tools import BaseTool
from pydantic import Field

class PDFParserTool(BaseTool):
    name = "pdf_parser"
    description = "Parse PDF content and extract structured information including title, abstract, sections, figures, and references"

    def _run(self, pdf_content: bytes) -> Dict:

        figures_pattern = re.compile(r'Figure \d+:.*?(?=Figure \d+:|$)', re.DOTALL)
        references_pattern = re.compile(r'References.*', re.DOTALL)
        
        """Parse PDF content and extract structured information"""
        pdf_file = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_file)
        
        # Extract text content
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()
        
        # Extract sections
        sections = self._extract_sections(full_text)
        
        # Extract figures
        figures = figures_pattern.findall(full_text) 
        
        # Extract references
        references_match = references_pattern.search(full_text)
        references = references_match.group() if references_match else ""
        
        return {
            'title': self._extract_title(full_text),
            'abstract': sections.get('Abstract', ''),
            'introduction': sections.get('Introduction', ''),
            'methods': sections.get('Methods', ''),
            'results': sections.get('Results', ''),
            'discussion': sections.get('Discussion', ''),
            'figures': figures,
            'references': references
        }
    
    async def _arun(self, pdf_content: bytes) -> Dict:
        """Async version of parse_pdf"""
        return self._run(pdf_content)
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract main sections from the paper"""
        sections = {}
        section_patterns = {
            'Abstract': r'Abstract.*?(?=Introduction|$)',
            'Introduction': r'Introduction.*?(?=Methods|Methodology|$)',
            'Methods': r'Methods.*?(?=Results|$)',
            'Results': r'Results.*?(?=Discussion|$)',
            'Discussion': r'Discussion.*?(?=Conclusion|References|$)'
        }
        
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[section] = match.group().strip()
        
        return sections
    
    def _extract_title(self, text: str) -> str:
        """Extract paper title"""
        first_line = text.split('\n')[0]
        return first_line.strip()