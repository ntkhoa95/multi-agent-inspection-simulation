# src/agents/diagnostic_agent.py
from datetime import datetime
from typing import Dict
from .base_agent import BaseAgent
from src.models.failure_predictor import FailurePredictor

class DiagnosticAgent(BaseAgent):
    def __init__(self, agent_id: str, config: Dict):
        super().__init__(agent_id, config)
        self.failure_predictor = FailurePredictor(
            input_size=config['sensor_dims'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        )

    async def handle_message(self, message: Dict):
        if message['type'] == 'anomaly_detected':
            diagnosis = await self._diagnose_issue(message)
            await self._request_maintenance(diagnosis)

    async def _diagnose_issue(self, anomaly_data: Dict) -> Dict:
        # Predict failure probability
        sensor_history = self._get_sensor_history(anomaly_data['equipment_id'])
        failure_prob = self._predict_failure(sensor_history)
        
        # Determine root cause
        root_cause = self._analyze_root_cause(anomaly_data['sensor_data'])
        
        return {
            'equipment_id': anomaly_data['equipment_id'],
            'timestamp': datetime.now(),
            'failure_probability': failure_prob,
            'root_cause': root_cause,
            'severity': self._calculate_severity(
                anomaly_data['anomaly_score'],
                failure_prob
            )
        }