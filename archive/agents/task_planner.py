from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from typing import List, Dict
import os

class TaskPlanner:
    def __init__(self):
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "gemma")
        self.llm = Ollama(
            base_url=ollama_host,
            model=ollama_model,
            temperature=0.3
        )
        self.prompt = ChatPromptTemplate.from_template(
            """Given a research topic or paper, create a detailed research plan.
            Break down the research into specific tasks and subtasks.
            
            Topic: {topic}
            
            Output format:
            1. Main Research Questions
            2. Required Information
            3. Specific Tasks
            4. Expected Outputs
            
            Create the plan:"""
        )
    
    async def create_plan(self, topic: str) -> Dict:
        """Create a research plan for the given topic"""
        response = await self.llm.agenerate([self.prompt.format_messages(topic=topic)])
        return self._parse_plan(response.generations[0][0].text)
    
    def _parse_plan(self, plan_text: str) -> Dict:
        """Parse the plan text into structured format"""
        sections = plan_text.split('\n\n')
        plan = {
            "research_questions": [],
            "required_info": [],
            "tasks": [],
            "expected_outputs": []
        }
        
        current_section = None
        for line in plan_text.split('\n'):
            if "Research Questions" in line:
                current_section = "research_questions"
            elif "Required Information" in line:
                current_section = "required_info"
            elif "Specific Tasks" in line:
                current_section = "tasks"
            elif "Expected Outputs" in line:
                current_section = "expected_outputs"
            elif line.strip() and current_section:
                plan[current_section].append(line.strip())
        
        return plan