# src/utils/data_generator.py
from typing import Dict, Optional
import numpy as np
from datetime import datetime, timedelta
import logging

class SensorDataGenerator:
    def __init__(self, config: Dict):
        """Initialize the sensor data generator with configuration."""
        self.config = config
        self.equipment_states = {}
        self.logger = logging.getLogger("SensorDataGenerator")
        
        # Set default values and override with config if available
        self.random_seed = config.get('simulation', {}).get('random_seed', 42)
        self.sensor_noise = config.get('equipment', {}).get('sensor_noise', 0.1)
        self.initial_health = config.get('equipment', {}).get('initial_health', 1.0)
        
        # Get degradation rate ranges
        degradation_config = config.get('equipment', {}).get('degradation_rates', {})
        self.min_degradation = degradation_config.get('min', 0.002)  # Increased base degradation
        self.max_degradation = degradation_config.get('max', 0.005)
        
        # Sensor baselines and thresholds
        sensor_config = config.get('sensors', {})
        self.vibration_baseline = sensor_config.get('vibration', {}).get('baseline', 1.0)
        self.temperature_baseline = sensor_config.get('temperature', {}).get('baseline', 40.0)
        self.pressure_baseline = sensor_config.get('pressure', {}).get('baseline', 100.0)
        
        # Operational parameters
        self.maintenance_threshold = config.get('maintenance_threshold', 0.7)
        self.critical_threshold = config.get('critical_threshold', 0.4)
        
        # Set random seed
        np.random.seed(self.random_seed)
        self.logger.info(f"Initialized SensorDataGenerator with seed {self.random_seed}")

    def initialize_equipment(self, equipment_id: str):
        """Initialize equipment state with default values."""
        self.equipment_states[equipment_id] = {
            'health': self.initial_health,
            'base_degradation_rate': np.random.uniform(self.min_degradation, self.max_degradation),
            'noise_level': self.sensor_noise,
            'last_maintenance': datetime.now(),
            'operating_hours': 0,
            'age_factor': 1.0,
            'stress_factor': 1.0,
            'maintenance_history': [],
            'failure_count': 0,
            'cumulative_stress': 0.0,
            'last_sudden_event': None
        }
        self.logger.info(f"Initialized equipment {equipment_id} with degradation rate: {self.equipment_states[equipment_id]['base_degradation_rate']:.6f}")

    def generate_sensor_data(self, equipment_id: str, timestamp: datetime) -> np.ndarray:
        """Generate sensor data for the specified equipment."""
        try:
            state = self.equipment_states[equipment_id]
            
            # Update equipment state
            time_delta = max(0, (timestamp - state['last_maintenance']).total_seconds() / 86400)  # days
            health = self._update_health(equipment_id, time_delta)
            
            # Calculate stress factors
            operating_stress = 1.0 + (state['operating_hours'] / 2000) * 0.2
            health_stress = 1.0 + max(0, (1 - health)) * 0.5
            
            # Update cumulative stress (ensure non-negative)
            stress_increment = max(0, (operating_stress * health_stress - 1) * 0.1)
            state['cumulative_stress'] = max(0, state['cumulative_stress'] + stress_increment)
            
            # Generate base sensor readings with stress influence
            vibration = self._generate_vibration(health) * (1 + state['cumulative_stress'] * 0.2)
            temperature = self._generate_temperature(health) * (1 + state['cumulative_stress'] * 0.15)
            pressure = self._generate_pressure(health) * max(0.1, 1 - state['cumulative_stress'] * 0.1)
            
            # Add noise that increases with degradation (ensure positive scale)
            noise_factor = max(self.sensor_noise * (1 + (1 - health) * 0.5), 1e-6)
            noise = np.random.normal(0, noise_factor, 3)
            
            readings = np.array([
                max(0, vibration + noise[0]),
                max(0, temperature + noise[1]),
                max(0, pressure + noise[2])
            ])
            
            # Update operating hours and log significant changes
            previous_hours = state['operating_hours']
            state['operating_hours'] += max(0, self.config.get('simulation', {}).get('time_step', 300) / 3600)
            
            if int(previous_hours / 1000) != int(state['operating_hours'] / 1000):
                self.logger.info(f"Equipment {equipment_id} reached {int(state['operating_hours']/1000)}k operating hours")
                
            return readings
            
        except Exception as e:
            self.logger.error(f"Error generating sensor data for {equipment_id}: {str(e)}")
            return np.array([self.vibration_baseline, self.temperature_baseline, self.pressure_baseline])

    def _update_health(self, equipment_id: str, time_delta: float) -> float:
        """Update and return equipment health based on time and degradation."""
        state = self.equipment_states[equipment_id]
        
        try:
            # Calculate age-based degradation acceleration
            age_factor = 1.0 + (state['operating_hours'] / 1000) * 0.15
            
            # Calculate stress-based degradation
            stress_factor = 1.0 + state['cumulative_stress']
            
            # Base degradation with factors
            base_degradation = (
                abs(state['base_degradation_rate']) *  # Ensure positive base rate
                time_delta * 
                abs(age_factor) *  # Ensure positive factors
                abs(stress_factor)
            )
            
            # Add random variations with absolute scale
            variation_scale = max(0.2 * base_degradation, 1e-6)  # Ensure positive scale with minimum value
            variation = np.random.normal(0, variation_scale)
            
            # Calculate total degradation (ensure positive)
            total_degradation = max(0, base_degradation + variation)
            
            # Add sudden degradation events
            current_time = datetime.now()
            time_since_last_event = float('inf') if state['last_sudden_event'] is None else (
                current_time - state['last_sudden_event']
            ).total_seconds()
            
            # Increase event probability with operating hours and low health
            event_probability = min(0.01 * (1 + state['operating_hours'] / 1000) * (1 + (1 - state['health'])), 0.05)
            
            if time_since_last_event > 3600 and np.random.random() < event_probability:
                sudden_degradation = np.random.uniform(0.05, 0.2) * (1 + state['cumulative_stress'])
                total_degradation += sudden_degradation
                state['last_sudden_event'] = current_time
                self.logger.warning(
                    f"Sudden degradation event for {equipment_id}: -{sudden_degradation:.3f} "
                    f"(Total stress: {state['cumulative_stress']:.2f})"
                )
            
            # Update health with bounds checking
            new_health = max(0.0, min(1.0, state['health'] - total_degradation))
            prev_health = state['health']
            state['health'] = new_health
            
            # Log significant health changes
            if prev_health - new_health > 0.1:
                self.logger.warning(f"Significant health drop for {equipment_id}: {prev_health:.3f} -> {new_health:.3f}")
            elif new_health < self.critical_threshold and prev_health >= self.critical_threshold:
                self.logger.warning(f"Equipment {equipment_id} entered critical state: {new_health:.3f}")
            
            return new_health
            
        except Exception as e:
            self.logger.error(f"Error updating health for {equipment_id}: {str(e)}")
            return state['health']  # Return current health in case of error

    def _generate_vibration(self, health: float) -> float:
        """Generate vibration reading based on health."""
        try:
            # Exponential increase in vibration as health decreases
            health = max(0.1, health)  # Prevent division by zero or negative values
            base_vibration = self.vibration_baseline * (1 + np.exp(1 - health))
            fluctuation = np.random.normal(0, max(0.1 * base_vibration, 1e-6))
            return max(0, base_vibration + fluctuation)
        except Exception as e:
            self.logger.error(f"Error generating vibration: {str(e)}")
            return self.vibration_baseline

    def _generate_temperature(self, health: float) -> float:
        """Generate temperature reading based on health."""
        try:
            # Non-linear temperature increase as health decreases
            health = max(0.1, health)  # Prevent negative values
            health_factor = 1 + (1 - health) ** 1.5
            base_temp = self.temperature_baseline * health_factor
            fluctuation = np.random.normal(0, max(2.0 * (2 - health), 1e-6))
            return max(0, base_temp + fluctuation)
        except Exception as e:
            self.logger.error(f"Error generating temperature: {str(e)}")
            return self.temperature_baseline

    def _generate_pressure(self, health: float) -> float:
        """Generate pressure reading based on health."""
        try:
            # Pressure decreases non-linearly with health
            health = max(0.1, health)  # Prevent division by zero
            health_factor = 0.5 + 0.5 * health ** 2
            base_pressure = self.pressure_baseline * health_factor
            fluctuation = np.random.normal(0, max(5.0 * (2 - health), 1e-6))
            return max(0, base_pressure + fluctuation)
        except Exception as e:
            self.logger.error(f"Error generating pressure: {str(e)}")
            return self.pressure_baseline

    def simulate_maintenance(self, equipment_id: str):
        """Simulate maintenance action on equipment."""
        if equipment_id not in self.equipment_states:
            self.logger.warning(f"Equipment {equipment_id} not found for maintenance")
            return

        state = self.equipment_states[equipment_id]
        
        # Record maintenance
        state['maintenance_history'].append({
            'timestamp': datetime.now(),
            'previous_health': state['health'],
            'operating_hours': state['operating_hours']
        })
        
        # Reset state
        state.update({
            'health': self.initial_health,
            'last_maintenance': datetime.now(),
            'cumulative_stress': max(0, state['cumulative_stress'] - 0.5),  # Reduce accumulated stress
            'stress_factor': 1.0
        })
        
        self.logger.info(
            f"Maintenance completed for {equipment_id}. "
            f"Total maintenance actions: {len(state['maintenance_history'])}"
        )

    def get_equipment_status(self, equipment_id: str) -> Optional[Dict]:
        """Get current status of specified equipment."""
        if equipment_id not in self.equipment_states:
            self.logger.warning(f"Equipment {equipment_id} not found")
            return None

        state = self.equipment_states[equipment_id]
        return {
            'health': state['health'],
            'operating_hours': state['operating_hours'],
            'time_since_maintenance': (
                datetime.now() - state['last_maintenance']
            ).total_seconds() / 3600,
            'degradation_rate': state['base_degradation_rate'] * state['age_factor'],
            'cumulative_stress': state['cumulative_stress'],
            'maintenance_count': len(state['maintenance_history'])
        }