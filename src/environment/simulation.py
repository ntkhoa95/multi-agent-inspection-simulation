# src/environment/simulation.py
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging
import pandas as pd
from ..utils.data_generator import SensorDataGenerator
from ..agents.maintenance_agent import MaintenanceAgent
from ..agents.base_agent import BaseAgent
from ..agents.monitor_agent import MonitorAgent


def setup_logging():
    """Configure logging for the simulation."""
    # Create formatters and handlers
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler('simulation.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Set levels for specific loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

# Call setup_logging() at the start of the simulation
setup_logging()

class MaintenanceSimulation:
    def __init__(self, config: Dict):
        self.config = config
        self.simulation_config = config.get('simulation', {})
        self.equipment_config = config.get('equipment', {})
        self.agents_config = config.get('agents', {})
        
        # Initialize components
        self.data_generator = SensorDataGenerator(config)
        self.agents = {}
        self.start_time = datetime.now()
        self.current_time = self.start_time
        self.history = []
        self.logger = self._setup_logger()
        
        # Define equipment groups - using full equipment IDs
        self.equipment_groups = {
            'mechanical': ['EQUIP_000', 'EQUIP_001'],
            'electrical': ['EQUIP_002', 'EQUIP_003'],
            'hydraulic': ['EQUIP_004']
        }

    async def initialize_simulation(self):
        """Initialize all simulation components in the correct order."""
        self.logger.info("Starting simulation initialization")
        
        # 1. Create maintenance agents first
        self._create_maintenance_agents()
        
        # 2. Initialize equipment and monitor agents
        self.initialize_equipment()
        
        # 3. Connect agents
        self._connect_agents()

        # 4. Validate connections
        self._validate_connections()
        
        self.logger.info("Simulation initialization completed")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("MaintenanceSimulation")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def initialize_equipment(self):
        """Initialize equipment and create monitor agents."""
        equipment_count = self.simulation_config.get('equipment_count', 5)
        self.logger.info(f"Initializing {equipment_count} equipment units")
        
        for i in range(equipment_count):
            equipment_id = f"EQUIP_{i:03d}"
            # Initialize equipment in data generator
            self.data_generator.initialize_equipment(equipment_id)
            
            # Create monitor agent
            monitor_agent = MonitorAgent(
                agent_id=f"monitor_{equipment_id}",
                equipment_id=equipment_id,
                config=self.agents_config
            )
            self.agents[f"monitor_{equipment_id}"] = monitor_agent
            self.logger.info(f"Created monitor agent for {equipment_id}")

    def _create_monitor_agent(self, equipment_id: str) -> MonitorAgent:
        """Create a monitor agent for the specified equipment."""
        agent_config = {
            **self.agents_config,
            'equipment_id': equipment_id,
            'sensor_config': self.config.get('sensors', {}),
            'health_threshold': 0.7,
            'critical_threshold': 0.4,
            'maintenance_cooldown': 3600,  # 1 hour in seconds
        }
        
        return MonitorAgent(
            agent_id=f"monitor_{equipment_id}",
            equipment_id=equipment_id,
            config=agent_config
        )

    def initialize_support_agents(self):
        """Initialize all support agents (diagnostic, planning, maintenance, etc.)."""
        self.logger.info("Initializing support agents")
        
        # Create maintenance agents (one per equipment group)
        maintenance_groups = {
            'mechanical': ['EQUIP_000', 'EQUIP_001'],
            'electrical': ['EQUIP_002', 'EQUIP_003'],
            'hydraulic': ['EQUIP_004']
        }
        
        for group, equipment_list in maintenance_groups.items():
            agent_id = f'maintenance_{group}'
            self.agents[agent_id] = MaintenanceAgent(
                agent_id=agent_id,
                config={
                    **self.agents_config,
                    'equipment_group': equipment_list
                }
            )
        
        # Connect agents
        self._connect_agents()

    def _connect_agents(self):
        """Establish connections between monitor and maintenance agents."""
        self.logger.info("Connecting agents")
        
        # Create a reverse mapping from equipment ID to group
        equipment_to_group = {}
        for group, equipment_list in self.equipment_groups.items():
            for equipment_id in equipment_list:
                equipment_to_group[equipment_id] = group
                self.logger.info(f"Mapped {equipment_id} to {group}")

        # Connect monitor agents to maintenance agents
        for agent_id, agent in self.agents.items():
            if agent_id.startswith('monitor_'):
                # Extract the full equipment ID (EQUIP_XXX) from the monitor agent ID
                equipment_id = agent_id.replace('monitor_', '')  # This will give us EQUIP_XXX
                
                # Find the corresponding group and maintenance agent
                group = equipment_to_group.get(equipment_id)
                if group:
                    maintenance_agent_id = f'maintenance_{group}'
                    if maintenance_agent_id in self.agents:
                        # Connect both ways
                        agent.connected_agents['maintenance'] = self.agents[maintenance_agent_id]
                        self.agents[maintenance_agent_id].connected_agents[agent_id] = agent
                        self.logger.info(f"Connected {agent_id} to {maintenance_agent_id}")
                    else:
                        self.logger.error(f"Maintenance agent {maintenance_agent_id} not found")
                else:
                    self.logger.error(f"No maintenance group found for {equipment_id}")

    async def run_simulation(self):
        """Run the main simulation loop."""
        self.logger.info("Starting simulation")
        
        # Initialize simulation components
        await self.initialize_simulation()
        
        # Start all agent tasks
        agent_tasks = []
        for agent_id, agent in self.agents.items():
            task = asyncio.create_task(agent.run())
            agent_tasks.append(task)
            self.logger.info(f"Started agent task: {agent_id}")
        
        # Run simulation steps
        sim_task = asyncio.create_task(self._simulation_loop())
        
        try:
            # Wait for simulation to complete
            await sim_task
            
            # Cancel agent tasks
            for task in agent_tasks:
                task.cancel()
            
            # Wait for all tasks to complete
            await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            self.logger.info("Simulation completed")
            return self.get_results()
            
        except Exception as e:
            self.logger.error(f"Error in simulation: {str(e)}", exc_info=True)
            raise

    async def _simulation_loop(self):
        """Main simulation loop."""
        time_step = self.simulation_config.get('time_step', 300)  # 5 minutes default
        duration_days = self.simulation_config.get('duration_days', 30)
        
        while (self.current_time - self.start_time).days < duration_days:
            # Generate and send sensor data for each equipment
            for equipment_id in self.data_generator.equipment_states.keys():
                sensor_data = self.data_generator.generate_sensor_data(
                    equipment_id,
                    self.current_time
                )
                
                await self._send_sensor_data(equipment_id, sensor_data)
            
            # Record simulation state
            self._record_state()
            
            # Advance time
            self.current_time += timedelta(seconds=time_step)
            await asyncio.sleep(0)  # Allow other tasks to run

    async def _send_sensor_data(self, equipment_id: str, sensor_data: np.ndarray):
        """Send sensor data to appropriate monitor agent."""
        monitor_agent = self.agents.get(f"monitor_{equipment_id}")
        if monitor_agent:
            message = {
                'type': 'sensor_data',
                'equipment_id': equipment_id,
                'timestamp': self.current_time,
                'data': sensor_data
            }
            await monitor_agent.message_queue.put(message)

    def _record_state(self):
        """Record current simulation state with minimal memory usage."""
        try:
            # Record only essential data
            state = {
                'timestamp': self.current_time,
                'equipment_states': {},
                'maintenance_actions': []
            }
            
            # Record equipment states (only health and key metrics)
            for eq_id in self.data_generator.equipment_states.keys():
                status = self.data_generator.get_equipment_status(eq_id)
                state['equipment_states'][eq_id] = {
                    'health': status['health'],
                    'operating_hours': status['operating_hours']
                }
            
            # Record only new maintenance actions
            for agent_id, agent in self.agents.items():
                if isinstance(agent, MaintenanceAgent):
                    for task in agent.active_tasks.values():
                        state['maintenance_actions'].append({
                            'agent_id': agent_id,
                            'equipment_id': task['action'].equipment_id,
                            'action_type': task['action'].action_type,
                            'timestamp': task['start_time']
                        })
            
            self.history.append(state)
            
            # Keep history size manageable
            if len(self.history) > 1000:  # Limit history size
                self.history = self.history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error recording state: {str(e)}")

    def get_results(self) -> Dict:
        """Get simulation results with proper timestamp formatting."""
        try:
            # Count maintenance actions from agents
            total_actions = 0
            maintenance_counts = {}
            action_types = set()
            
            # Collect maintenance actions with proper timestamps
            for agent_id, agent in self.agents.items():
                if isinstance(agent, MaintenanceAgent):
                    agent_actions = []
                    for eq_id, history in agent.maintenance_history.items():
                        total_actions += len(history)
                        for record in history:
                            action_types.add(record['action_type'])
                            action = {
                                'timestamp': record['timestamp'],
                                'action_type': record['action_type'],
                                'equipment_id': eq_id,
                                'success': record.get('success', True)
                            }
                            agent_actions.append(action)
                    
                    # Sort actions by timestamp
                    agent_actions.sort(key=lambda x: x['timestamp'])
                    maintenance_counts[agent_id] = agent_actions

            # Calculate average health
            current_health = [
                state['health']
                for state in self.data_generator.equipment_states.values()
            ]
            avg_health = sum(current_health) / len(current_health) if current_health else 0

            results = {
                'simulation_duration': self.current_time - self.start_time,
                'start_time': self.start_time,
                'end_time': self.current_time,
                'total_maintenance_actions': total_actions,
                'maintenance_counts': maintenance_counts,
                'metrics': {
                    'average_health': avg_health,
                    'action_types': len(action_types),
                    'actions_per_day': total_actions / max((self.current_time - self.start_time).days, 1)
                }
            }

            self.logger.info(f"Collected results: {total_actions} maintenance actions")
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting results: {str(e)}")
            return {
                'simulation_duration': self.current_time - self.start_time,
                'start_time': self.start_time,
                'end_time': self.current_time,
                'total_maintenance_actions': 0,
                'maintenance_counts': {},
                'metrics': {
                    'average_health': 0,
                    'action_types': 0,
                    'actions_per_day': 0
                }
            }

    @property
    def simulation_duration(self):
        """Get simulation duration."""
        return self.current_time - self.start_time

    def _summarize_agent_actions(self) -> Dict:
        """Summarize agent actions from history."""
        try:
            summaries = {}
            for agent_id, agent in self.agents.items():
                if isinstance(agent, MaintenanceAgent):
                    actions = []
                    for equipment_id, history in agent.maintenance_history.items():
                        for record in history:
                            actions.append({
                                'timestamp': record['timestamp'],
                                'equipment_id': equipment_id,
                                'action_type': record['action_type'],
                                'success': record['success']
                            })
                    summaries[agent_id] = actions
                    
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error summarizing agent actions: {str(e)}", exc_info=True)
            return {}
    
    def _create_maintenance_agents(self):
        """Create maintenance agents for each equipment group."""
        self.logger.info("Creating maintenance agents")
        
        for group, equipment_list in self.equipment_groups.items():
            agent_id = f'maintenance_{group}'
            agent_config = {
                **self.agents_config,
                'equipment_group': equipment_list,
                'group_type': group
            }
            
            maintenance_agent = MaintenanceAgent(
                agent_id=agent_id,
                config=agent_config
            )
            self.agents[agent_id] = maintenance_agent
            self.logger.info(f"Created maintenance agent: {agent_id} for equipment {equipment_list}")

    async def simulate_maintenance(self, equipment_id: str):
        """Simulate maintenance effect on equipment."""
        try:
            # Reset equipment health
            self.data_generator.simulate_maintenance(equipment_id)
            
            # Record maintenance effect
            status = self.data_generator.get_equipment_status(equipment_id)
            maintenance_record = {
                'timestamp': self.current_time,
                'equipment_id': equipment_id,
                'action': 'maintenance',
                'old_health': status.get('health', 0),
                'new_health': 1.0  # Reset to full health
            }
            
            self.logger.info(f"Maintenance completed for {equipment_id}: Health restored to 1.0")
            return maintenance_record
            
        except Exception as e:
            self.logger.error(f"Error simulating maintenance: {str(e)}", exc_info=True)
            return None
    
    def _validate_connections(self):
        """Validate that all agents are properly connected."""
        self.logger.info("Validating agent connections")
        
        # Check monitor agents
        for agent_id, agent in self.agents.items():
            if agent_id.startswith('monitor_'):
                equipment_id = agent_id.replace('monitor_', '')  # Get EQUIP_XXX
                if 'maintenance' not in agent.connected_agents:
                    self.logger.error(f"Monitor agent {agent_id} has no maintenance connection")
                else:
                    maintenance_agent = agent.connected_agents['maintenance']
                    self.logger.info(f"Validated connection: {agent_id} -> {maintenance_agent.agent_id}")
        
        # Check maintenance agents
        for agent_id, agent in self.agents.items():
            if agent_id.startswith('maintenance_'):
                group = agent_id.split('_')[1]
                expected_equipment = self.equipment_groups[group]
                connected_equipment = [
                    conn_id.replace('monitor_', '') for conn_id in agent.connected_agents.keys()
                    if conn_id.startswith('monitor_')
                ]
                self.logger.info(f"Maintenance agent {agent_id} connections: {connected_equipment}")
                
                for equipment_id in expected_equipment:
                    if f"monitor_{equipment_id}" not in agent.connected_agents:
                        self.logger.error(f"Missing connection for {equipment_id} in {agent_id}")