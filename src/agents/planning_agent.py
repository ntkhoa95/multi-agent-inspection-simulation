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
        self.connected_agents = {}

    async def handle_message(self, message: Dict):
        if message['type'] == 'maintenance_needed':
            await self._plan_maintenance(message)
        elif message['type'] == 'resource_update':
            self.resource_manager.update(message['resources'])
        elif message['type'] == 'maintenance_complete':
            await self._update_maintenance_plan(message)

    async def _plan_maintenance(self, maintenance_request: Dict):
        # Create maintenance task
        task = self._create_task(maintenance_request)
        
        # Schedule task
        schedule = self.scheduler.schedule_task(task)
        
        # Allocate resources
        resources = self.resource_manager.allocate(task)
        
        # Notify connected agents
        await self._notify_agents(task, schedule, resources)

    def _create_task(self, maintenance_request: Dict):
        # Create a maintenance task based on the request
        task = {
            'equipment_id': maintenance_request['equipment_id'],
            'priority': maintenance_request['priority'],
            'action_type': maintenance_request['action_type'],
            'estimated_duration': maintenance_request['estimated_duration'],
            'resources_required': maintenance_request['resources_required']
        }
        return task

    async def _notify_agents(self, task: Dict, schedule: Dict, resources: Dict):
        # Notify connected agents about the planned maintenance task
        for agent_id, agent in self.connected_agents.items():
            message = {
                'type': 'maintenance_planned',
                'task': task,
                'schedule': schedule,
                'resources': resources
            }
            await agent.message_queue.put(message)

    async def _update_maintenance_plan(self, maintenance_complete: Dict):
        # Update the maintenance plan based on completed tasks
        equipment_id = maintenance_complete['equipment_id']
        
        # Remove the completed task from the task queue
        for task in self.task_queue:
            if task['equipment_id'] == equipment_id:
                self.task_queue.remove(task)
                break
        
        # Free up the resources used for the completed task
        self.resource_manager.free_resources(maintenance_complete['resources'])
        
        # Reschedule any pending tasks that were affected by the freed resources
        self._reschedule_tasks()
        
        # Notify connected agents about the updated maintenance plan
        await self._notify_agents_of_plan_update()

    def _reschedule_tasks(self):
        # Reschedule any pending tasks based on the updated resource availability
        for task in self.task_queue:
            schedule = self.scheduler.schedule_task(task)
            resources = self.resource_manager.allocate(task)
            # Update the task's schedule and resource allocation
            task['schedule'] = schedule
            task['resources'] = resources

    async def _notify_agents_of_plan_update(self):
        # Notify connected agents about the updated maintenance plan
        for agent_id, agent in self.connected_agents.items():
            message = {
                'type': 'maintenance_plan_updated',
                'tasks': self.task_queue
            }
            await agent.message_queue.put(message)
