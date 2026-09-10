from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from typing import Dict, List, Any
from pydantic import Field, PrivateAttr
import os

class SummarizerTool(BaseTool):
    name = "summarizer"
    description = "Generate summaries and extract key points from research papers and text content"
    _llm: Any = PrivateAttr()
    _summary_prompt: Any = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Get Ollama configuration from environment variables
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'gemma')
        
        # Using Ollama with configuration from environment variables
        object.__setattr__(self, '_llm', Ollama(
            base_url=ollama_host,
            model=ollama_model
        ))
        object.__setattr__(self, '_summary_prompt', ChatPromptTemplate.from_template(
            """Summarize the following research paper content:
            
            {content}
            
            Provide a structured summary including:
            1. Main findings
            2. Methodology
            3. Key contributions
            4. Limitations
            5. Future work suggestions
            """
        ))
    
    def _run(self, text: str) -> str:
        """Synchronous version of summarize (required by BaseTool)"""
        raise NotImplementedError("This tool only supports async execution")
    
    async def _arun(self, text: str) -> str:
        """Generate a summary of the given text"""
        prompt = f"""
        Please provide a concise summary of the following text.
        Focus on the main points, key findings, and conclusions.
        
        Text to summarize:
        {text}
        
        Summary:
        """
        
        result = await self._llm.agenerate([prompt])
        return result.generations[0][0].text.strip()
    
    async def extract_key_points(self, text: str) -> Dict:
        """Extract key points from the text"""
        prompt = f"""
        Extract the key points from the following text. Format the response as a JSON object with these fields:
        - main_topic: The main topic or research question
        - methodology: The research methodology used
        - key_findings: List of main findings
        - conclusions: Main conclusions
        - implications: Implications for future research
        
        Text:
        {text}
        """
        
        result = await self._llm.agenerate([prompt])
        return {
            "summary": result.generations[0][0].text.strip(),
            "key_points": self._parse_key_points(result.generations[0][0].text)
        }
    
    def _parse_key_points(self, text: str) -> Dict:
        """Parse the key points from the LLM response"""
        # Simple parsing logic - can be improved based on actual response format
        sections = text.split('\n')
        key_points = {
            "main_topic": "",
            "methodology": "",
            "key_findings": [],
            "conclusions": "",
            "implications": ""
        }
        
        current_section = None
        for line in sections:
            line = line.strip()
            if not line:
                continue
                
            if "main_topic" in line.lower():
                current_section = "main_topic"
            elif "methodology" in line.lower():
                current_section = "methodology"
            elif "key_findings" in line.lower():
                current_section = "key_findings"
            elif "conclusions" in line.lower():
                current_section = "conclusions"
            elif "implications" in line.lower():
                current_section = "implications"
            elif current_section:
                if current_section == "key_findings":
                    key_points[current_section].append(line)
                else:
                    key_points[current_section] = line
        
        return key_points
    
    async def summarize_paper(self, paper_content: Dict) -> Dict:
        """Generate a structured summary of a paper"""
        # Combine relevant sections
        content = f"""
        Abstract: {paper_content.get('abstract', '')}
        
        Introduction: {paper_content.get('introduction', '')}
        
        Methods: {paper_content.get('methods', '')}
        
        Results: {paper_content.get('results', '')}
        
        Discussion: {paper_content.get('discussion', '')}
        """
        
        response = await self._llm.agenerate([
            self._summary_prompt.format_messages(content=content)
        ])
        
        return self._parse_summary(response.generations[0][0].text)
    
    async def compare_papers(self, summaries: List[Dict]) -> str:
        """Compare multiple paper summaries"""
        comparison_prompt = f"""Compare the following research papers:
        
        {self._format_summaries(summaries)}
        
        Provide:
        1. Common themes
        2. Key differences
        3. Complementary findings
        4. Research gaps
        5. Synthesis of insights
        """
        
        response = await self._llm.agenerate([comparison_prompt])
        return response.generations[0][0].text
    
    def _parse_summary(self, summary_text: str) -> Dict:
        """Parse the summary text into structured format"""
        sections = {
            'main_findings': [],
            'methodology': [],
            'key_contributions': [],
            'limitations': [],
            'future_work': []
        }
        
        current_section = None
        for line in summary_text.split('\n'):
            if 'Main findings' in line:
                current_section = 'main_findings'
            elif 'Methodology' in line:
                current_section = 'methodology'
            elif 'Key contributions' in line:
                current_section = 'key_contributions'
            elif 'Limitations' in line:
                current_section = 'limitations'
            elif 'Future work' in line:
                current_section = 'future_work'
            elif line.strip() and current_section:
                sections[current_section].append(line.strip())
        
        return sections
    
    def _format_summaries(self, summaries: List[Dict]) -> str:
        """Format multiple summaries for comparison"""
        formatted = ""
        for i, summary in enumerate(summaries, 1):
            formatted += f"\nPaper {i}:\n"
            for section, points in summary.items():
                formatted += f"\n{section.replace('_', ' ').title()}:\n"
                formatted += "\n".join(f"- {point}" for point in points)
            formatted += "\n---\n"
        return formatted