# src/agents/planning_agent.py
from typing import Dict
from .base_agent import BaseAgent
from ..models.scheduler import MaintenanceScheduler
from ..utils.resource_manager import ResourceManager

class PlanningAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        self.scheduler = MaintenanceScheduler(config)
        self.task_queue = []
        self.resource_manager = ResourceManager(config)

    async def handle_message(self, message: Dict):
        if message['type'] == 'maintenance_needed':
            await self._plan_maintenance(message)
        elif message['type'] == 'resource_update':
            self.resource_manager.update(message['resources'])

    async def _plan_maintenance(self, maintenance_request: Dict):
        # Create maintenance task
        task = self._create_task(maintenance_request)
        
        # Schedule task
        schedule = self.scheduler.schedule_task(task)
        
        # Allocate resources
        resources = self.resource_manager.allocate(task)
        
        # Notify coordination agent
        await self._notify_coordination(task, schedule, resources)