# src/agents/equipment_monitor.py
from typing import Dict, List
import torch
import numpy as np
from datetime import datetime
from .base_agent import BaseAgent
from ..models.anomaly_detector import AnomalyDetector

class EquipmentMonitorAgent(BaseAgent):
    def __init__(self, agent_id: str, equipment_id: str, config: Dict):
        super().__init__(agent_id, config)
        self.equipment_id = equipment_id
        self.anomaly_detector = AnomalyDetector(
            input_size=config['sensor_dims'],
            hidden_size=config['hidden_size']
        )
        self.normal_behavior = self._initialize_normal_behavior()
        self.anomaly_threshold = config['anomaly_threshold']

    async def handle_message(self, message: Dict):
        if message['type'] == 'sensor_data':
            await self._process_sensor_data(message['data'])
        elif message['type'] == 'update_threshold':
            self.anomaly_threshold = message['threshold']

    async def _process_sensor_data(self, sensor_data: np.ndarray):
        # Convert to tensor
        data_tensor = torch.FloatTensor(sensor_data)
        
        # Get anomaly score
        anomaly_score = self.anomaly_detector.get_anomaly_score(data_tensor)
        
        # Check for anomalies
        if anomaly_score > self.anomaly_threshold:
            await self._report_anomaly(sensor_data, anomaly_score)

        # Update normal behavior model
        self._update_normal_behavior(sensor_data, anomaly_score)

    async def _report_anomaly(self, sensor_data: np.ndarray, anomaly_score: float):
        message = {
            'type': 'anomaly_detected',
            'equipment_id': self.equipment_id,
            'timestamp': datetime.now(),
            'sensor_data': sensor_data,
            'anomaly_score': anomaly_score
        }
        
        for agent_id, agent in self.connected_agents.items():
            if agent_id.startswith('diagnostic'):
                await self.send_message(agent, message)