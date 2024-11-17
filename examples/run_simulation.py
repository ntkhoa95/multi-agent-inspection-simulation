# examples/run_simulation.py
import asyncio
import yaml
import logging
from datetime import datetime
from src.environment.simulation import MaintenanceSimulation
from src.utils.visualization import plot_simulation_results

async def main():
    try:
        # Load configuration
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # Create and run simulation
        sim = MaintenanceSimulation(config)
        results = await sim.run_simulation()

        # Print summary
        print("\nSimulation Results:")
        print(f"Duration: {results['simulation_duration']}")
        print(f"Total Maintenance Actions: {results['total_maintenance_actions']}")
        
        # Print maintenance actions by type
        print("\nMaintenance Actions by Type:")
        action_types = {}
        for agent_id, actions in results['maintenance_counts'].items():
            for action in actions:
                action_type = action['action_type']
                if action_type not in action_types:
                    action_types[action_type] = 0
                action_types[action_type] += 1
        
        for action_type, count in action_types.items():
            print(f"{action_type}: {count}")

        # Plot results
        plot_simulation_results(results)

    except Exception as e:
        logging.error(f"Error in simulation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())