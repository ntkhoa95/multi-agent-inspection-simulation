# src/agents/monitor_agent.py
from typing import Dict, Optional
import numpy as np
from datetime import datetime
import logging, asyncio
from .base_agent import BaseAgent

# src/agents/monitor_agent.py
from typing import Dict, Optional
import numpy as np
from datetime import datetime
import logging
from .base_agent import BaseAgent

class MonitorAgent(BaseAgent):
    def __init__(self, agent_id: str, equipment_id: str, config: Dict):
        super().__init__(agent_id, config)
        self.equipment_id = equipment_id
        self.health_threshold = config.get('health_threshold', 0.6)
        self.critical_threshold = config.get('critical_threshold', 0.3)
        self.warning_threshold = config.get('warning_threshold', 0.7)
        self.last_maintenance_request = None
        self.maintenance_cooldown = config.get('maintenance_cooldown', 3600)
        self.health_history = []
        self.alert_history = []
        self.degradation_rate = None
        self.logger = logging.getLogger(f"MonitorAgent_{equipment_id}")

        # Get sensor baselines from config
        sensor_config = config.get('sensors', {})
        self.vibration_baseline = sensor_config.get('vibration', {}).get('baseline', 1.0)
        self.temperature_baseline = sensor_config.get('temperature', {}).get('baseline', 40.0)
        self.pressure_baseline = sensor_config.get('pressure', {}).get('baseline', 100.0)

    def _calculate_health_score(self, sensor_data: np.ndarray) -> float:
        """Calculate health score from sensor data."""
        try:
            # Validate sensor data
            if not isinstance(sensor_data, np.ndarray) or sensor_data.size != 3:
                self.logger.error("Invalid sensor data format")
                return 0.0

            # Get sensor readings
            vibration = abs(sensor_data[0])
            temperature = abs(sensor_data[1])
            pressure = abs(sensor_data[2])
            
            # Calculate normalized deviations
            try:
                vibration_health = max(0, 1 - (vibration - self.vibration_baseline) / max(self.vibration_baseline, 1e-6))
                temp_health = max(0, 1 - abs(temperature - self.temperature_baseline) / max(self.temperature_baseline, 1e-6))
                pressure_health = max(0, 1 - abs(pressure - self.pressure_baseline) / max(self.pressure_baseline, 1e-6))
            except ZeroDivisionError:
                self.logger.error("Zero baseline value encountered")
                return 0.0
            
            # Weighted average
            health_score = (
                0.4 * vibration_health +
                0.3 * temp_health +
                0.3 * pressure_health
            )
            
            # Log detailed health components
            self.logger.debug(
                f"Health components for {self.equipment_id}: "
                f"vibration={vibration_health:.3f}, "
                f"temperature={temp_health:.3f}, "
                f"pressure={pressure_health:.3f}, "
                f"total={health_score:.3f}"
            )
            
            return max(0.0, min(1.0, health_score))
        
        except Exception as e:
            self.logger.error(f"Error calculating health score: {str(e)}")
            return 0.0

    async def _process_sensor_data(self, message: Dict):
        """Process sensor data and determine if maintenance is needed."""
        try:
            # Extract and validate data
            sensor_data = message.get('data')
            if not isinstance(sensor_data, np.ndarray):
                self.logger.error("Invalid sensor data type")
                return

            timestamp = message.get('timestamp', datetime.now())
            
            # Calculate health score
            current_health = self._calculate_health_score(sensor_data)
            self.logger.debug(f"Calculated health score: {current_health:.3f}")
            
            # Update health history
            history_entry = {
                'timestamp': timestamp,
                'health': current_health,
                'sensor_data': sensor_data.tolist()
            }
            self.health_history.append(history_entry)
            
            # Keep only recent history
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]
            
            # Calculate degradation rate
            self._update_degradation_rate()
            
            # Check if maintenance is needed
            await self._check_maintenance_needs(current_health, timestamp)
            
            # Update agent state
            self.state.update({
                'current_health': current_health,
                'degradation_rate': self.degradation_rate,
                'last_reading': sensor_data.tolist(),
                'last_update': timestamp.isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error processing sensor data: {str(e)}")
            raise

    def _update_degradation_rate(self):
        """Calculate equipment degradation rate with better logging."""
        try:
            if len(self.health_history) < 2:
                return

            recent_health = [h['health'] for h in self.health_history[-10:]]
            timestamps = [h['timestamp'] for h in self.health_history[-10:]]
            
            if not recent_health or not timestamps:
                return
            
            time_diff = (timestamps[-1] - timestamps[0]).total_seconds()
            if time_diff > 0:
                health_diff = recent_health[-1] - recent_health[0]
                self.degradation_rate = health_diff / time_diff
                
                if abs(self.degradation_rate) > 0.1:
                    self.logger.warning(
                        f"High degradation rate detected for {self.equipment_id}: "
                        f"{self.degradation_rate:.3f} per second"
                    )
                
        except Exception as e:
            self.logger.error(f"Error calculating degradation rate: {str(e)}")
            self.degradation_rate = None

    async def handle_message(self, message: Dict):
        """Handle incoming messages."""
        try:
            message_type = message.get('type')
            self.logger.debug(f"Handling message type: {message_type}")
            
            if message_type == 'sensor_data':
                await self._process_sensor_data(message)
                self.logger.debug(f"Processed sensor data: health={self.state.get('current_health', 0):.3f}")
            elif message_type == 'maintenance_complete':
                await self._handle_maintenance_complete(message)
                self.logger.info(f"Handled maintenance completion for {self.equipment_id}")
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")

    async def _check_maintenance_needs(self, current_health: float, timestamp: datetime):
        """Determine if maintenance is needed with proper cooldown handling."""
        try:
            # Skip checks if health is still good
            if current_health > 0.7:
                return

            # Calculate cooldown elapsed time
            cooldown_elapsed = float('inf')  # Default to infinite time if no last request
            if self.last_maintenance_request:
                cooldown_elapsed = (timestamp - self.last_maintenance_request).total_seconds()

            # Base cooldown periods
            CRITICAL_COOLDOWN = self.maintenance_cooldown * 2
            WARNING_COOLDOWN = self.maintenance_cooldown * 3
            PREVENTIVE_COOLDOWN = self.maintenance_cooldown * 4

            maintenance_needed = False
            priority = 0.0
            reason = ""

            # Check conditions with proper cooldown handling
            if current_health < self.critical_threshold:  # Critical: health < 0.3
                if cooldown_elapsed > CRITICAL_COOLDOWN:
                    maintenance_needed = True
                    priority = 1.0
                    reason = f"CRITICAL: Health at {current_health:.2f}"
                    self.logger.warning(
                        f"Critical condition for {self.equipment_id}: "
                        f"Health={current_health:.2f}, Time since last maintenance={cooldown_elapsed/3600:.1f}h"
                    )
            
            elif current_health < self.health_threshold:  # Warning: health < 0.5
                if cooldown_elapsed > WARNING_COOLDOWN:
                    # Only if degrading significantly
                    if self.degradation_rate and self.degradation_rate < -0.1:
                        maintenance_needed = True
                        priority = 0.7
                        reason = f"WARNING: Health below threshold at {current_health:.2f}"
                        self.logger.info(
                            f"Warning condition for {self.equipment_id}: "
                            f"Health={current_health:.2f}, Degradation={self.degradation_rate:.3f}"
                        )
            
            elif self.degradation_rate and self.degradation_rate < -0.2:  # Preventive
                if cooldown_elapsed > PREVENTIVE_COOLDOWN:
                    maintenance_needed = True
                    priority = 0.5
                    reason = f"PREVENTIVE: Severe degradation (rate: {self.degradation_rate:.3f})"
                    self.logger.info(
                        f"Preventive maintenance needed for {self.equipment_id}: "
                        f"Degradation={self.degradation_rate:.3f}"
                    )

            if maintenance_needed:
                # Log maintenance decision
                self.logger.info(
                    f"Requesting maintenance for {self.equipment_id}:\n"
                    f"  Health: {current_health:.2f}\n"
                    f"  Degradation Rate: {self.degradation_rate if self.degradation_rate else 'N/A'}\n"
                    f"  Time Since Last Maintenance: {cooldown_elapsed/3600:.1f}h\n"
                    f"  Priority: {priority}\n"
                    f"  Reason: {reason}"
                )
                
                success = await self._request_maintenance(current_health, priority, reason)
                if success:
                    self.last_maintenance_request = timestamp
                    self.logger.info(f"Maintenance request successful: {reason}")
                else:
                    self.logger.warning(f"Maintenance request failed for {self.equipment_id}")

        except Exception as e:
            self.logger.error(f"Error checking maintenance needs: {str(e)}", exc_info=True)

    async def _request_maintenance(self, health: float, priority: float, reason: str) -> bool:
        """Send maintenance request with better logging."""
        try:
            if 'maintenance' not in self.connected_agents:
                self.logger.error(f"No maintenance agent connected for {self.equipment_id}")
                return False

            maintenance_agent = self.connected_agents['maintenance']
            
            # Prepare request with detailed information
            request = {
                'type': 'maintenance_request',
                'equipment_id': self.equipment_id,
                'timestamp': datetime.now(),
                'priority': priority,
                'reason': reason,
                'equipment_data': {
                    'health': health,
                    'degradation_rate': self.degradation_rate,
                    'recent_history': self.health_history[-10:] if self.health_history else [],
                    'sensor_data': self.state.get('last_reading', [])
                }
            }
            
            # Send request
            await self.send_message(maintenance_agent, request)
            
            # Log request details
            self.logger.info(
                f"Maintenance request sent:\n"
                f"  Equipment: {self.equipment_id}\n"
                f"  Agent: {maintenance_agent.agent_id}\n"
                f"  Priority: {priority}\n"
                f"  Reason: {reason}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending maintenance request: {str(e)}", exc_info=True)
            return False

    async def _handle_maintenance_complete(self, message: Dict):
        """Handle maintenance completion notification."""
        try:
            self.logger.info(f"Processing maintenance completion")
            self.last_maintenance_request = None
            
            # Reset health in history
            if self.health_history:
                self.health_history[-1]['health'] = 1.0
            
            # Update state
            self.state.update({
                'last_maintenance': datetime.now().isoformat(),
                'maintenance_completed': True
            })
            
            self.logger.info("Maintenance completion processed")
            
        except Exception as e:
            self.logger.error(f"Error handling maintenance completion: {str(e)}")