# Multi-Agent Maintenance System API Documentation

## Agent Classes

### `MaintenanceAgent`
**Purpose**: The `MaintenanceAgent` class is responsible for executing maintenance actions, recording maintenance history, and optimizing maintenance strategies.

**Methods:**
- `handle_message(message: Dict)`: Handles incoming messages, such as maintenance requests.
- `_execute_maintenance_action(action: MaintenanceAction)`: Executes a maintenance action and returns the success and reward.
- `_create_state(equipment_id: str, equipment_data: Dict, resource_data: Dict)`: Creates a `MaintenanceState` object based on the current data.
- `_create_maintenance_action(action_idx: int, equipment_id: str, state: MaintenanceState)`: Creates a `MaintenanceAction` based on the current state and a selected action.

**Usage Example:**
```python
maintenance_agent = MaintenanceAgent(agent_id='maintenance_1', config=config)
maintenance_request = {
    'equipment_id': 'EQUIP_001',
    'priority': 0.8,
    'action_type': 'repair',
    'estimated_duration': 2.0,
    'resources_required': ['technician', 'tools']
}
success, reward = await maintenance_agent._execute_maintenance_action(maintenance_request)
```

### `MonitorAgent`
**Purpose**: The `MonitorAgent` class monitors the health and status of individual equipment units, triggering maintenance requests as needed.

**Methods:**
- `handle_message(message: Dict)`: Handles incoming messages, such as sensor data.
- `_process_sensor_data(equipment_id: str, sensor_data: np.ndarray)`: Processes the sensor data and updates the equipment's health and failure probability.
- `_trigger_maintenance_request(equipment_id: str, priority: float, reason: str)`: Triggers a maintenance request for the specified equipment.

**Usage Example:**
```python
monitor_agent = MonitorAgent(agent_id='monitor_EQUIP_001', equipment_id='EQUIP_001', config=config)
sensor_data = np.array([...])  # Sensor data for the equipment
await monitor_agent._process_sensor_data('EQUIP_001', sensor_data)
if monitor_agent.equipment_health < 0.7:
    await monitor_agent._trigger_maintenance_request('EQUIP_001', 0.8, 'Low equipment health')
```

### `KnowledgeAgent`, `DiagnosticAgent`, `PlanningAgent`, and others
The documentation for these agent classes follows a similar structure, providing an overview of their responsibilities and the key methods they implement.

## Simulation Classes

### `MaintenanceSimulation`
**Purpose**: The `MaintenanceSimulation` class orchestrates the overall simulation, including initializing the agents, connecting them, and running the main simulation loop.

**Methods:**
- `initialize_simulation()`: Initializes all simulation components in the correct order.
- `run_simulation()`: Runs the main simulation loop, generating sensor data and recording the state.
- `get_results()`: Collects and formats the simulation results.

**Usage Example:**
```python
simulation = MaintenanceSimulation(config)
await simulation.run_simulation()
results = simulation.get_results()
print(results)
```

### `SensorDataGenerator`
**Purpose**: The `SensorDataGenerator` class is responsible for generating realistic sensor data for the simulated equipment.

**Methods:**
- `initialize_equipment(equipment_id: str)`: Initializes the equipment state.
- `generate_sensor_data(equipment_id: str, timestamp: datetime)`: Generates sensor data for the specified equipment at the given timestamp.

**Usage Example:**
```python
data_generator = SensorDataGenerator(config)
data_generator.initialize_equipment('EQUIP_001')
sensor_data = data_generator.generate_sensor_data('EQUIP_001', datetime.now())
```

## Model Classes

### `FailurePredictor`
**Purpose**: The `FailurePredictor` class predicts the likelihood of equipment failures based on sensor data and historical maintenance records.

**Methods:**
- `forward(x: torch.Tensor)`: Performs a forward pass through the model to predict the failure probability.

**Usage Example:**
```python
failure_predictor = FailurePredictor(input_size=10, hidden_size=64, num_layers=2)
failure_probability = failure_predictor.forward(sensor_data)
```

### `AnomalyDetector`
**Purpose**: The `AnomalyDetector` class identifies anomalies in equipment behavior that may indicate potential issues.

**Methods:**
- `forward(x: torch.Tensor)`: Performs a forward pass through the model to detect anomalies.

**Usage Example:**
```python
anomaly_detector = AnomalyDetector(input_size=10, hidden_size=32)
is_anomaly = anomaly_detector.forward(sensor_data)
```

### `Scheduler`
**Purpose**: The `Scheduler` class optimizes the maintenance schedule based on factors like cost, resources, and priority.

**Methods:**
- `schedule_maintenance(state: MaintenanceState, actions: List[MaintenanceAction])`: Generates an optimized maintenance schedule.

**Usage Example:**
```python
scheduler = Scheduler(config)
maintenance_state = maintenance_agent._create_state(...)
maintenance_actions = [
    maintenance_agent._create_maintenance_action(0, 'EQUIP_001', maintenance_state),
    maintenance_agent._create_maintenance_action(1, 'EQUIP_002', maintenance_state)
]
schedule = scheduler.schedule_maintenance(maintenance_state, maintenance_actions)
```

## Utility Functions and Classes

### `ResourceManager`
**Purpose**: The `ResourceManager` class manages the allocation and availability of resources (e.g., technicians, tools, parts) required for maintenance activities.

**Methods:**
- `allocate(task: MaintenanceAction)`: Allocates the necessary resources for a given maintenance task.
- `free_resources(resources: Dict)`: Frees up the resources used for a completed maintenance task.
- `update(resources: Dict)`: Updates the resource availability information.

**Usage Example:**
```python
resource_manager = ResourceManager(config)
resources = resource_manager.allocate(maintenance_action)
# Perform maintenance
resource_manager.free_resources(resources)
```

### `Visualization` tools
**Purpose**: The `Visualization` module provides various functions and classes for generating visualizations related to the multi-agent maintenance system, such as equipment health trends, maintenance activity charts, and resource utilization graphs.

**Functions:**
- `plot_equipment_health(equipment_states: Dict)`: Generates a line plot of equipment health over time.
- `plot_maintenance_activities(maintenance_history: Dict)`: Creates a bar chart of maintenance activities by type and agent.
- `plot_resource_utilization(resource_usage: Dict)`: Visualizes the utilization of different resources over time.

**Usage Example:**
```python
from src.utils.visualization import plot_equipment_health

equipment_states = simulation.get_equipment_states()
plot_equipment_health(equipment_states)
```

This detailed API documentation should provide users with a clear understanding of the various components of the multi-agent maintenance system and how they can be used to achieve the system's objectives.
