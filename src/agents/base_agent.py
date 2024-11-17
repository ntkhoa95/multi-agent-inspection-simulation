# src/agents/base_agent.py
from typing import Dict, List, Optional, Any
import asyncio
import logging
from datetime import datetime
import numpy as np

class BaseAgent:
    def __init__(self, agent_id: str, config: Dict):
        self.agent_id = agent_id
        self.config = config
        self.message_queue = asyncio.Queue()
        self.state = {}
        self.knowledge_base = {}
        self.connected_agents = {}
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"Agent_{self.agent_id}")
        logger.setLevel(logging.DEBUG)
        return logger

    async def send_message(self, target_agent: 'BaseAgent', message: Dict):
        """Send message to another agent."""
        try:
            message['sender_id'] = self.agent_id
            message['timestamp'] = datetime.now()
            await target_agent.message_queue.put(message)
            self.logger.debug(f"Sent message to {target_agent.agent_id}: {message['type']}")
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")

    async def receive_message(self) -> Dict:
        message = await self.message_queue.get()
        self.logger.debug(f"Received message from {message['sender']}: {message['type']}")
        return message

    async def run(self):
        """Main agent loop."""
        self.logger.info(f"Starting agent {self.agent_id}")
        try:
            while True:
                message = await self.message_queue.get()
                self.logger.debug(f"Received message: {message['type']} from {message.get('sender_id')}")
                await self.process_message(message)
                self.message_queue.task_done()
        except Exception as e:
            self.logger.error(f"Error in agent loop: {str(e)}", exc_info=True)

    async def process_message(self, message: Dict):
        """Process incoming message."""
        try:
            await self.handle_message(message)
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", exc_info=True)

    async def handle_message(self, message: Dict):
        """Handle message - to be implemented by derived classes."""
        raise NotImplementedError("handle_message must be implemented by derived classes")

    def update_state(self, new_state: Dict):
        self.state.update(new_state)
        self.logger.debug(f"State updated: {new_state}")

    def learn(self, experience: Dict):
        pass