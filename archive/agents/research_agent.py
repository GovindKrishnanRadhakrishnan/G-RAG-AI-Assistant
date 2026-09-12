from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.tools import BaseTool
from typing import List, Dict, Any
import os
from ..tools.paper_search import PaperSearchTool
from ..tools.pdf_parser import PDFParserTool
from ..tools.summarizer import SummarizerTool

class CustomPromptTemplate(PromptTemplate):
    def format_prompt(self, **kwargs) -> str:
        # Get the intermediate string from PromptTemplate
        prompt = self.template.format(**kwargs)
        return prompt

    def format(self, **kwargs) -> str:
        return self.format_prompt(**kwargs)

class ResearchAgent:
    def __init__(self):
        # Get Ollama configuration from environment variables
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        ollama_model = os.getenv('OLLAMA_MODEL', 'gemma')
        
        # Using Ollama with configuration from environment variables
        self.llm = Ollama(
            base_url=ollama_host,
            model=ollama_model
        )
        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()
    
    def _setup_tools(self) -> List[BaseTool]:
        return [
            PaperSearchTool(),
            PDFParserTool(),
            SummarizerTool()
        ]
    
    def _setup_agent(self) -> AgentExecutor:
        prompt = """You are a research assistant. Given a research topic or paper,
        your task is to:
        1. Find relevant papers
        2. Analyze their content
        3. Summarize key findings
        4. Compare different approaches
        5. Suggest future research directions

        Current task: {input}
        Available tools: {tools}
        Tool names: {tool_names}

        Think step by step:
        1) What information do I need?
        2) Which tool should I use?
        3) How should I process the results?

        Response should be in this format:
        Thought: your thought process
        Action: tool_name
        Action Input: input to the tool
        Observation: result of action
        ... (repeat until task is complete)
        Final Answer: your final response

        {agent_scratchpad}

        Begin!
        """
        
        prompt_template = PromptTemplate(
            template=prompt,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt_template
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True
        )
    
    async def research(self, topic: str) -> Dict:
        """Execute research on a given topic"""
        result = await self.agent_executor.arun(topic)
        return {
            "topic": topic,
            "findings": result,
            "sources": self.agent_executor.tools[0].get_sources()
        }
    
    async def analyze_paper(self, content: bytes) -> Dict:
        """Analyze a research paper"""
        # Parse the PDF
        pdf_parser = PDFParserTool()
        # parsed_content = pdf_parser._run(content)
        parsed_content = pdf_parser._run(content)
        
        # Generate analysis using the LLM
        analysis_prompt = f"""
        Analyze this research paper and provide:
        1. Main research question
        2. Methodology used
        3. Key findings
        4. Limitations
        5. Future research directions

        Paper content:
        Title: {parsed_content['title']}
        Abstract: {parsed_content['abstract']}
        Introduction: {parsed_content['introduction']}
        Methods: {parsed_content['methods']}
        Results: {parsed_content['results']}
        Discussion: {parsed_content['discussion']}
        """
        
        analysis = await self.llm.agenerate([analysis_prompt])
        return {
            "title": parsed_content['title'],
            "analysis": analysis.generations[0][0].text,
            "figures": parsed_content['figures'],
            "references": parsed_content['references']
        }