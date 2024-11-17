# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict
import logging

# Disable matplotlib debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

def plot_simulation_results(results: Dict):
    """Plot simulation results with improved layout."""
    plt.style.use('seaborn')
    
    # Create figure with adjusted size and spacing
    fig = plt.figure(figsize=(16, 12))
    gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1], 
                     width_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # System Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    plot_metrics(ax1, results['metrics'])
    
    # Maintenance Actions by Agent
    ax2 = fig.add_subplot(gs[0, 1])
    plot_maintenance_by_agent(ax2, results['maintenance_counts'])
    
    # Maintenance Timeline (full width)
    ax3 = fig.add_subplot(gs[1, :])
    plot_maintenance_timeline(ax3, results['maintenance_counts'])
    
    # Summary Statistics (full width)
    ax4 = fig.add_subplot(gs[2, :])
    plot_summary_stats(ax4, results)
    
    # plt.tight_layout()
    plt.show()

def plot_metrics(ax, metrics):
    """Plot key metrics."""
    metrics_list = [
        ('Average Health', metrics['average_health']),
        ('Action Types', metrics['action_types']),
        ('Actions/Day', metrics['actions_per_day'])
    ]
    
    y_pos = range(len(metrics_list))
    values = [m[1] for m in metrics_list]
    labels = [m[0] for m in metrics_list]
    
    bars = ax.barh(y_pos, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(v, i, f' {v:.2f}')
    
    ax.set_title('System Metrics')

def plot_maintenance_by_agent(ax, maintenance_counts):
    """Plot maintenance actions by agent."""
    if not maintenance_counts:
        ax.text(0.5, 0.5, 'No maintenance data', ha='center', va='center')
        return
        
    agents = list(maintenance_counts.keys())
    counts = [len(actions) for actions in maintenance_counts.values()]
    
    y_pos = range(len(agents))
    ax.barh(y_pos, counts)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agents)
    
    ax.set_title('Maintenance Actions by Agent')
    ax.set_xlabel('Number of Actions')

def plot_maintenance_timeline(ax, maintenance_counts):
    """Plot maintenance actions over time with proper timeline visualization."""
    try:
        if not maintenance_counts:
            ax.text(0.5, 0.5, 'No maintenance data', ha='center', va='center')
            return

        # Create color map for different agents
        agents = list(maintenance_counts.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(agents)))
        
        # Plot maintenance events for each agent
        for (agent_id, actions), color in zip(maintenance_counts.items(), colors):
            if actions:
                # Convert timestamps to matplotlib dates
                timestamps = [action['timestamp'] for action in actions]
                y_position = agents.index(agent_id)  # Stacked position for each agent
                
                # Plot maintenance events as vertical lines
                ax.vlines(timestamps, 
                         y_position - 0.3, y_position + 0.3,  # Line length
                         color=color, alpha=0.6, label=agent_id)
                
                # Add markers at the maintenance points
                ax.plot(timestamps, [y_position] * len(timestamps), 
                       'o', color=color, markersize=4, alpha=0.5)

        # Customize the plot
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents)
        
        # Format x-axis to show dates properly
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_title('Maintenance Timeline')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting timeline: {str(e)}',
                ha='center', va='center')

def plot_summary_stats(ax, results):
    """Plot summary statistics."""
    stats = [
        ('Total Actions', results['total_maintenance_actions']),
        ('Duration (days)', results['simulation_duration'].days),
        ('Actions/Day', results['metrics']['actions_per_day'])
    ]
    
    y_pos = range(len(stats))
    values = [s[1] for s in stats]
    labels = [s[0] for s in stats]
    
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    
    ax.set_title('Summary Statistics')

def plot_equipment_health(ax, results):
    """Plot equipment health over time with maintenance markers."""
    try:
        # Extract data
        timestamps = [state['timestamp'] for state in results['equipment_states']]
        equipment_ids = list(results['equipment_states'][0]['states'].keys())
        
        # Plot health for each equipment
        for eq_id in equipment_ids:
            health_values = [state['states'][eq_id]['health'] 
                           for state in results['equipment_states']]
            ax.plot(timestamps, health_values, 
                   label=f'Equipment {eq_id}',
                   marker='.', markersize=4)

        # Add maintenance markers
        if results['maintenance_actions']:
            maintenance_times = [action['start_time'] 
                               for action in results['maintenance_actions']]
            ax.vlines(maintenance_times, 0, 1, 
                     colors='green', alpha=0.2, linestyles='--')

        ax.set_ylim(0, 1.1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Health Score')
        ax.set_title('Equipment Health Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting health: {str(e)}',
                ha='center', va='center')
        ax.set_title('Equipment Health Plot (Error)')

def plot_maintenance_actions(ax, results):
    """Plot maintenance action distribution."""
    try:
        if not results['maintenance_actions']:
            ax.text(0.5, 0.5, 'No maintenance actions recorded',
                   ha='center', va='center')
            return

        # Plot histogram of maintenance times
        maintenance_times = [action['start_time'] 
                           for action in results['maintenance_actions']]
        counts, bins, patches = ax.hist(maintenance_times, bins=20, 
                                      alpha=0.7, color='blue')

        # Add action type labels
        for action in results['maintenance_actions']:
            y_pos = counts.max() * 0.1
            ax.text(action['start_time'], y_pos,
                   action['action_type'],
                   rotation=45, ha='right',
                   fontsize=8, alpha=0.5)

        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Actions')
        ax.set_title('Maintenance Actions Distribution')
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting maintenance: {str(e)}',
                ha='center', va='center')

def plot_agent_activity(ax, results):
    """Plot agent activity summary."""
    try:
        # Count actions per agent
        agent_counts = {}
        for action in results['maintenance_actions']:
            agent_id = action['agent_id']
            if agent_id not in agent_counts:
                agent_counts[agent_id] = {'total': 0, 'types': {}}
            
            agent_counts[agent_id]['total'] += 1
            
            action_type = action['action_type']
            if action_type not in agent_counts[agent_id]['types']:
                agent_counts[agent_id]['types'][action_type] = 0
            agent_counts[agent_id]['types'][action_type] += 1

        if agent_counts:
            agents = list(agent_counts.keys())
            totals = [agent_counts[agent]['total'] for agent in agents]
            
            # Create stacked bars for different action types
            bottom = np.zeros(len(agents))
            action_types = sorted(set(
                action_type
                for agent_data in agent_counts.values()
                for action_type in agent_data['types'].keys()
            ))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(action_types)))
            
            for action_type, color in zip(action_types, colors):
                values = [agent_counts[agent]['types'].get(action_type, 0) 
                         for agent in agents]
                ax.barh(agents, values, left=bottom, 
                       label=action_type, color=color, alpha=0.7)
                bottom += values

            ax.set_xlabel('Number of Actions')
            ax.set_title('Agent Activity Summary')
            ax.legend(title='Action Types', bbox_to_anchor=(1.05, 1), 
                     loc='upper left')
            
        else:
            ax.text(0.5, 0.5, 'No agent activities recorded',
                   ha='center', va='center')
            
    except Exception as e:
        ax.text(0.5, 0.5, f'Error plotting activities: {str(e)}',
                ha='center', va='center')

def plot_performance_metrics(ax, results):
    """Plot system performance metrics."""
    # Calculate metrics
    total_actions = results['total_maintenance_actions']
    
    # Calculate average health properly
    health_values = [
        state['health']
        for equipment_state in results['equipment_states']
        for state in equipment_state['states'].values()
        if isinstance(state.get('health'), (int, float))  # Add type check
    ]
    avg_health = np.mean(health_values) if health_values else 0
    
    # Count unique action types
    action_types = set(action['action_type'] for action in results['maintenance_actions'])
    num_action_types = len(action_types)
    
    # Calculate daily action rate
    simulation_days = max(results['simulation_duration'].days, 1)
    actions_per_day = total_actions / simulation_days
    
    metrics = {
        'Total Actions': total_actions,
        'Average Health': round(avg_health, 3),
        'Action Types': num_action_types,
        'Actions/Day': round(actions_per_day, 2)
    }
    
    # Plot bars with custom colors
    x_pos = range(len(metrics))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(metrics)))
    bars = ax.bar(x_pos, list(metrics.values()), color=colors, alpha=0.7)
    
    # Add value labels
    for bar, value in zip(bars, metrics.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}' if isinstance(value, float) else str(value),
                ha='center', va='bottom')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics.keys(), rotation=45)
    ax.set_title('System Performance Metrics')
    ax.grid(True, alpha=0.3)

def calculate_performance_metrics(results):
    """Calculate various performance metrics from simulation results."""
    try:
        # Extract equipment states
        equipment_states = results['equipment_health_history']['equipment_states'].values
        
        # Calculate average health
        avg_health = np.mean([
            np.mean([state[eq_id]['health'] 
                    for eq_id in state.keys()])
            for state in equipment_states
        ])
        
        # Calculate maintenance frequency
        maintenance_count = results['total_maintenance_actions']
        simulation_days = results['simulation_duration'].days
        maintenance_frequency = maintenance_count / max(simulation_days, 1)
        
        return {
            'Average Health': avg_health,
            'Maintenance Actions/Day': maintenance_frequency,
            'Total Maintenance Actions': maintenance_count
        }
        
    except Exception as e:
        return {
            'Error': 0,
            'Message': str(e)
        }

# Example usage
if __name__ == "__main__":
    # Example results structure
    example_results = {
        'simulation_duration': timedelta(days=30),
        'total_maintenance_actions': 150,
        'equipment_health_history': pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'equipment_states': [
                {'EQUIP_001': {'health': 0.9}, 'EQUIP_002': {'health': 0.85}}
                for _ in range(100)
            ]
        }),
        'agent_actions': {
            'maintenance_mechanical': [
                {'timestamp': datetime(2024, 1, 1, 12, 0), 'action': 'repair'}
                for _ in range(10)
            ],
            'monitor_EQUIP_001': [
                {'timestamp': datetime(2024, 1, 1, 12, 0), 'action': 'measure'}
                for _ in range(20)
            ]
        }
    }
    
    # Plot results
    plot_simulation_results(example_results)