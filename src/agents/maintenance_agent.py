# src/agents/maintenance_agent.py
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from .base_agent import BaseAgent
from ..models.learning import LearningModule, Experience
import logging
import asyncio

@dataclass
class MaintenanceAction:
    action_type: str  # 'inspect', 'repair', 'replace', 'adjust'
    equipment_id: str
    priority: float
    estimated_duration: float
    resources_required: List[str]
    expected_improvement: float

@dataclass
class MaintenanceState:
    equipment_health: float
    time_since_last_maintenance: float
    failure_probability: float
    resource_availability: Dict[str, float]
    workload: float
    priority_score: float

class MaintenanceAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        
        # Initialize learning module
        learning_config = {
            'state_dim': 6,  # Matches MaintenanceState attributes
            'action_dim': 4,  # Number of possible maintenance actions
            'hidden_dim': 64,
            'learning_rate': config.get('learning_rate', 0.001),
            'memory_size': config.get('memory_size', 10000),
            'batch_size': config.get('batch_size', 64),
            'gamma': config.get('gamma', 0.99),
            'tau': config.get('tau', 0.005),
            'epsilon_start': config.get('epsilon_start', 1.0),
            'epsilon_end': config.get('epsilon_end', 0.01),
            'epsilon_decay': config.get('epsilon_decay', 0.995),
            'success_threshold': config.get('success_threshold', 0.7)
        }
        self.learning_module = LearningModule(learning_config)
        
        # Initialize agent-specific attributes
        self.equipment_group = config.get('equipment_group', [])
        self.group_type = config.get('group_type', 'unknown')
        self.active_tasks = {}
        self.maintenance_history = {}
        
        # Performance metrics
        self.performance_metrics = {
            'successful_actions': 0,
            'failed_actions': 0,
            'total_downtime': 0,
            'total_cost': 0,
            'maintenance_counts': {}
        }
        
        # Action mappings with parameters
        self.action_params = {
            'inspect': {
                'duration': 1.0,
                'resources': ['technician'],
                'improvement': 0.2,
                'cost': 100
            },
            'repair': {
                'duration': 2.0,
                'resources': ['technician', 'tools'],
                'improvement': 0.5,
                'cost': 500
            },
            'replace': {
                'duration': 4.0,
                'resources': ['technician', 'tools', 'parts'],
                'improvement': 1.0,
                'cost': 1000
            },
            'adjust': {
                'duration': 1.5,
                'resources': ['technician'],
                'improvement': 0.3,
                'cost': 200
            }
        }
        
        self.action_map = {
            0: 'inspect',
            1: 'repair',
            2: 'replace',
            3: 'adjust'
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"MaintenanceAgent_{agent_id}")
        self.logger.setLevel(logging.DEBUG)

    async def handle_message(self, message: Dict):
        """Handle incoming messages."""
        try:
            message_type = message.get('type')
            self.logger.debug(f"Received message type: {message_type} from {message.get('sender_id')}")
            
            if message_type == 'maintenance_request':
                self.logger.info(f"Processing maintenance request for {message.get('equipment_id')}")
                await self._handle_maintenance_request(message)
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}", exc_info=True)

    async def _handle_maintenance_request(self, message: Dict):
        """Handle maintenance request from monitor agent."""
        try:
            equipment_id = message['equipment_id']
            priority = message['priority']
            reason = message['reason']
            
            self.logger.info(f"Processing maintenance request for {equipment_id} (Priority: {priority})")
            self.logger.info(f"Reason: {reason}")
            
            # Create maintenance action
            action = MaintenanceAction(
                action_type='replace' if priority > 0.8 else 'repair',
                equipment_id=equipment_id,
                priority=priority,
                estimated_duration=2.0,  # hours
                resources_required=['technician', 'tools'],
                expected_improvement=0.8 if priority > 0.8 else 0.5
            )
            
            # Execute maintenance
            success, reward = await self._execute_maintenance_action(action)
            
            if success:
                self.logger.info(f"Successfully completed maintenance for {equipment_id}")
            else:
                self.logger.error(f"Failed to complete maintenance for {equipment_id}")
                
        except Exception as e:
            self.logger.error(f"Error handling maintenance request: {str(e)}", exc_info=True)

    async def _execute_maintenance_action(self, action: MaintenanceAction) -> Tuple[bool, float]:
        """Execute a maintenance action and return success and reward."""
        try:
            task_id = f"TASK_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.info(f"Starting maintenance task {task_id} for {action.equipment_id}")
            
            # Record task start
            self.active_tasks[task_id] = {
                'action': action,
                'start_time': datetime.now(),
                'status': 'in_progress'
            }
            
            # Initialize maintenance history for equipment if needed
            if action.equipment_id not in self.maintenance_history:
                self.maintenance_history[action.equipment_id] = []
            
            # Record maintenance action
            maintenance_record = {
                'task_id': task_id,
                'timestamp': datetime.now(),
                'action_type': action.action_type,
                'priority': action.priority,
                'success': True
            }
            self.maintenance_history[action.equipment_id].append(maintenance_record)
            
            # Notify monitor agent
            await self._notify_maintenance_completion(action.equipment_id)
            
            # Calculate reward
            reward = self._calculate_reward(action)
            
            # Update metrics
            self.performance_metrics['successful_actions'] += 1
            
            self.logger.info(f"Completed maintenance task {task_id}")
            return True, reward
            
        except Exception as e:
            self.logger.error(f"Error executing maintenance: {str(e)}")
            return False, -1.0

    async def _notify_monitor(self, equipment_id: str, task_id: str):
        """Notify monitor agent of maintenance completion."""
        try:
            monitor_agent_id = f"monitor_{equipment_id}"
            if monitor_agent_id in self.connected_agents:
                message = {
                    'type': 'maintenance_complete',
                    'equipment_id': equipment_id,
                    'task_id': task_id,
                    'timestamp': datetime.now(),
                    'maintenance_type': self.group_type
                }
                await self.send_message(self.connected_agents[monitor_agent_id], message)
                self.logger.info(f"Notified {monitor_agent_id} of maintenance completion")
        except Exception as e:
            self.logger.error(f"Error notifying monitor: {str(e)}")

    async def _notify_maintenance_completion(self, equipment_id: str):
        """Notify monitor agent of maintenance completion."""
        monitor_agent_id = f"monitor_{equipment_id}"
        if monitor_agent_id in self.connected_agents:
            message = {
                'type': 'maintenance_complete',
                'equipment_id': equipment_id,
                'timestamp': datetime.now(),
                'maintenance_type': self.group_type
            }
            await self.send_message(self.connected_agents[monitor_agent_id], message)
            self.logger.info(f"Notified {monitor_agent_id} of maintenance completion")

    def _create_state(self, equipment_id: str, equipment_data: Dict, 
                     resource_data: Dict) -> MaintenanceState:
        """Create maintenance state from current data."""
        last_maintenance = self.maintenance_history.get(equipment_id, [])
        if last_maintenance:
            last_maintenance_time = last_maintenance[-1]['timestamp']
        else:
            last_maintenance_time = datetime.now() - timedelta(days=30)
        
        time_since_maintenance = (
            datetime.now() - last_maintenance_time
        ).total_seconds() / 86400  # Convert to days
        
        return MaintenanceState(
            equipment_health=equipment_data.get('health', 1.0),
            time_since_last_maintenance=time_since_maintenance,
            failure_probability=equipment_data.get('failure_prob', 0.0),
            resource_availability=resource_data.get('availability', {}),
            workload=len(self.active_tasks),
            priority_score=equipment_data.get('priority', 0.0)
        )

    def _create_maintenance_action(self, action_idx: int, equipment_id: str, 
                                 state: MaintenanceState) -> MaintenanceAction:
        """Create maintenance action based on state and selected action."""
        action_type = self.action_map[action_idx]
        
        # If health is critical, upgrade the action
        if state.equipment_health < 0.5:
            if action_type == 'inspect':
                action_type = 'repair'
            elif action_type == 'repair':
                action_type = 'replace'
        
        params = self.action_params[action_type]
        
        return MaintenanceAction(
            action_type=action_type,
            equipment_id=equipment_id,
            priority=state.priority_score,
            estimated_duration=params['duration'],
            resources_required=params['resources'].copy(),
            expected_improvement=params['improvement']
        )

    def _record_maintenance(self, equipment_id: str, task_id: str, 
                          action: MaintenanceAction):
        """Record maintenance in history."""
        if equipment_id not in self.maintenance_history:
            self.maintenance_history[equipment_id] = []
            
        completion_record = {
            'task_id': task_id,
            'action_type': action.action_type,
            'timestamp': datetime.now(),
            'duration': action.estimated_duration,
            'resources': action.resources_required.copy(),
            'success': True
        }
        
        self.maintenance_history[equipment_id].append(completion_record)
        self.logger.info(f"Recorded maintenance for {equipment_id}: {completion_record}")

    def _update_state(self, task_id: str, status: str):
        """Update agent state with task information."""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            self.state.update({
                'last_task': {
                    'task_id': task_id,
                    'equipment_id': task['action'].equipment_id,
                    'action_type': task['action'].action_type,
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                }
            })

    def _calculate_reward(self, action: MaintenanceAction) -> float:
        """Calculate reward for maintenance action."""
        base_reward = self.action_params[action.action_type]['improvement']
        priority_factor = action.priority
        return base_reward * priority_factor

    def _update_metrics(self, success: bool, action: MaintenanceAction):
        """Update performance metrics after maintenance."""
        if success:
            self.performance_metrics['successful_actions'] += 1
            
            # Update maintenance counts
            if action.equipment_id not in self.performance_metrics['maintenance_counts']:
                self.performance_metrics['maintenance_counts'][action.equipment_id] = 0
            self.performance_metrics['maintenance_counts'][action.equipment_id] += 1
            
            # Update costs
            base_cost = self.action_params[action.action_type]['cost']
            resource_cost = len(action.resources_required) * 50
            self.performance_metrics['total_cost'] += (base_cost + resource_cost)
        else:
            self.performance_metrics['failed_actions'] += 1

    def get_performance_report(self) -> Dict:
        """Generate performance report."""
        total_actions = (self.performance_metrics['successful_actions'] + 
                        self.performance_metrics['failed_actions'])
        
        success_rate = (self.performance_metrics['successful_actions'] / 
                       max(total_actions, 1) * 100)
        
        return {
            'success_rate': success_rate,
            'total_actions': total_actions,
            'average_cost': (self.performance_metrics['total_cost'] / 
                           max(total_actions, 1)),
            'active_tasks': len(self.active_tasks),
            'maintenance_counts': self.performance_metrics['maintenance_counts'],
            'learning_stats': self.learning_module.extract_insights([])
        }