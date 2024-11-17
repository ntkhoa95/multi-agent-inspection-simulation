# src/agents/coordination_agent.py
from typing import Dict
from .base_agent import BaseAgent

class CoordinationAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        self.active_tasks = {}
        self.team_status = {}
        self.maintenance_history = []

    async def handle_message(self, message: Dict):
        if message['type'] == 'task_scheduled':
            await self._coordinate_maintenance(message)
        elif message['type'] == 'task_complete':
            await self._process_completion(message)

    async def _coordinate_maintenance(self, schedule_info: Dict):
        # Assign maintenance teams
        team_assignments = self._assign_teams(schedule_info)
        
        # Update task status
        self._update_task_status(schedule_info, team_assignments)
        
        # Notify assigned teams
        await self._notify_teams(schedule_info, team_assignments)