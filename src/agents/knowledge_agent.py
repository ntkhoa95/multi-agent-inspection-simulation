# src/agents/knowledge_agent.py
from typing import Dict
from .base_agent import BaseAgent
from ..models.learning import LearningModule

class KnowledgeAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        self.knowledge_base = {}
        self.learning_module = LearningModule(config)

    async def handle_message(self, message: Dict):
        if message['type'] == 'maintenance_completed':
            await self._process_feedback(message)
        elif message['type'] == 'knowledge_request':
            await self._provide_knowledge(message)

    async def _process_feedback(self, feedback: Dict):
        # Update knowledge base
        self._update_knowledge(feedback)
        
        # Learn from experience
        self.learning_module.learn(feedback)
        
        # Share insights with other agents
        await self._share_insights(feedback)