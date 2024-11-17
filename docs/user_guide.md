# Multi-Agent Maintenance System User Guide

## Overview
The multi-agent maintenance system is a powerful tool for simulating the coordination and optimization of maintenance activities across a fleet of equipment. The system utilizes a collection of specialized agents, including maintenance agents, monitor agents, and various support agents, to monitor equipment health, plan and execute maintenance actions, and optimize the overall maintenance strategy.

## Running the Simulation
To run the simulation, follow these steps:

1. Set up the development environment as described in the project's [README.md](../README.md).
2. Customize the configuration settings in `config/config.yaml` as needed.
3. Execute the example script `examples/run_simulation.py` to start the simulation.

The simulation will run for the specified duration (default is 30 days) and generate a `simulation.log` file with detailed logs of the agents' activities and the overall system performance.

## Interpreting the Results
The simulation results can be found in the `simulation.log` file. The key metrics reported include:

- **Total Maintenance Actions**: The total number of maintenance actions performed during the simulation.
- **Maintenance Counts**: The number of each type of maintenance action (inspect, repair, replace, adjust) performed by each agent.
- **Average Health**: The average health of the equipment units across the simulation.
- **Actions per Day**: The average number of maintenance actions performed per day.

Use these metrics to understand the system's performance and effectiveness in maintaining the equipment fleet.

## Troubleshooting
If you encounter any issues or errors while running the simulation, please check the `simulation.log` file for more information. Common issues and their solutions include:

- **Missing Dependencies**: Ensure you have installed all the required dependencies by running `pip install -r requirements.txt`.
- **Configuration Issues**: Verify that the settings in `config/config.yaml` are correct and consistent with your environment.
- **Unexpected Behavior**: If the simulation is not behaving as expected, please report the issue on the project's GitHub repository, providing as much detail as possible.

## Reporting Issues and Requesting Features
If you encounter any bugs or have feature requests, please open an issue on the project's GitHub repository. Provide a clear description of the problem or the desired feature, along with any relevant information or logs.

## Contribution Guidelines
We welcome contributions to the project. Please refer to the [Contribution Guidelines](../README.md#contribution-guidelines) section in the project's README.md file for more information.
