# src/utils/resource_manager.py
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from collections import defaultdict
import asyncio
from queue import PriorityQueue

@dataclass
class Resource:
    id: str
    type: str  # 'technician', 'tool', 'part', 'equipment'
    status: str  # 'available', 'assigned', 'maintenance', 'unavailable'
    capabilities: List[str]
    current_assignment: Optional[str]  # task_id if assigned
    availability_schedule: Dict[str, List[Tuple[datetime, datetime]]]
    efficiency_rating: float
    maintenance_history: List[Dict]
    cost_per_hour: float

@dataclass
class ResourceRequest:
    request_id: str
    task_id: str
    resource_type: str
    quantity: int
    priority: float
    start_time: datetime
    duration: timedelta
    capabilities_required: List[str]

class ResourceManager:
    def __init__(self, config: Dict):
        self.config = config
        self.resources: Dict[str, Resource] = {}
        self.resource_pools: Dict[str, List[str]] = defaultdict(list)
        self.assignments: Dict[str, Dict] = {}
        self.pending_requests = PriorityQueue()
        self.scheduled_releases = PriorityQueue()
        self.logger = logging.getLogger("ResourceManager")
        
        # Initialize resources from config
        self._initialize_resources()

    def _initialize_resources(self):
        """Initialize resources from configuration."""
        for resource_config in self.config.get('resources', []):
            resource = Resource(
                id=resource_config['id'],
                type=resource_config['type'],
                status='available',
                capabilities=resource_config.get('capabilities', []),
                current_assignment=None,
                availability_schedule=resource_config.get('schedule', {}),
                efficiency_rating=resource_config.get('efficiency', 1.0),
                maintenance_history=[],
                cost_per_hour=resource_config.get('cost_per_hour', 0.0)
            )
            
            self.resources[resource.id] = resource
            self.resource_pools[resource.type].append(resource.id)

    async def request_resources(self, request: ResourceRequest) -> Tuple[bool, List[str]]:
        """Request resources for a task."""
        try:
            # Check immediate availability
            available_resources = self._find_available_resources(request)
            
            if len(available_resources) >= request.quantity:
                # Sufficient resources available
                assigned_resources = available_resources[:request.quantity]
                await self._assign_resources(assigned_resources, request)
                return True, assigned_resources
            
            # If not immediately available, queue the request
            await self._queue_request(request)
            return False, []
            
        except Exception as e:
            self.logger.error(f"Error requesting resources: {str(e)}")
            return False, []

    def _find_available_resources(self, request: ResourceRequest) -> List[str]:
        """Find available resources matching request criteria."""
        available = []
        
        for resource_id in self.resource_pools[request.resource_type]:
            resource = self.resources[resource_id]
            
            if self._is_resource_available(resource, request):
                available.append(resource_id)
        
        # Sort by efficiency rating
        available.sort(
            key=lambda r: self.resources[r].efficiency_rating,
            reverse=True
        )
        
        return available

    def _is_resource_available(self, resource: Resource, 
                             request: ResourceRequest) -> bool:
        """Check if a resource is available for the requested time period."""
        if resource.status != 'available':
            return False
            
        # Check capabilities
        if not all(cap in resource.capabilities 
                  for cap in request.capabilities_required):
            return False
            
        # Check schedule
        request_period = (request.start_time, 
                        request.start_time + request.duration)
        
        for scheduled_start, scheduled_end in resource.availability_schedule.get(
            request.start_time.strftime('%Y-%m-%d'), []
        ):
            if (request_period[0] >= scheduled_start and 
                request_period[1] <= scheduled_end):
                return True
                
        return False

    async def _assign_resources(self, resource_ids: List[str], 
                              request: ResourceRequest):
        """Assign resources to a task."""
        assignment = {
            'task_id': request.task_id,
            'start_time': request.start_time,
            'duration': request.duration,
            'resources': resource_ids
        }
        
        self.assignments[request.task_id] = assignment
        
        # Update resource status
        for resource_id in resource_ids:
            resource = self.resources[resource_id]
            resource.status = 'assigned'
            resource.current_assignment = request.task_id
            
        # Schedule resource release
        release_time = request.start_time + request.duration
        await self._schedule_release(request.task_id, release_time)
        
        self.logger.info(
            f"Assigned resources {resource_ids} to task {request.task_id}"
        )

    async def _queue_request(self, request: ResourceRequest):
        """Queue a resource request for later fulfillment."""
        await self.pending_requests.put(
            (
                -request.priority,  # Negative for priority queue (highest first)
                request
            )
        )
        self.logger.info(f"Queued resource request for task {request.task_id}")

    async def _schedule_release(self, task_id: str, release_time: datetime):
        """Schedule resource release."""
        await self.scheduled_releases.put((release_time, task_id))

    async def release_resources(self, task_id: str):
        """Release resources from a task."""
        if task_id not in self.assignments:
            self.logger.warning(f"No assignment found for task {task_id}")
            return
            
        assignment = self.assignments[task_id]
        
        # Update resource status
        for resource_id in assignment['resources']:
            resource = self.resources[resource_id]
            resource.status = 'available'
            resource.current_assignment = None
            
            # Update maintenance history
            resource.maintenance_history.append({
                'task_id': task_id,
                'start_time': assignment['start_time'],
                'duration': assignment['duration']
            })
            
        del self.assignments[task_id]
        self.logger.info(f"Released resources from task {task_id}")
        
        # Process pending requests
        await self._process_pending_requests()

    async def _process_pending_requests(self):
        """Process pending resource requests."""
        while not self.pending_requests.empty():
            _, request = await self.pending_requests.get()
            
            success, resources = await self.request_resources(request)
            if not success:
                # If still can't fulfill, put back in queue
                await self._queue_request(request)
                break

    def get_resource_status(self) -> Dict[str, Dict]:
        """Get current status of all resources."""
        status = defaultdict(list)
        
        for resource in self.resources.values():
            status[resource.type].append({
                'id': resource.id,
                'status': resource.status,
                'current_assignment': resource.current_assignment,
                'efficiency_rating': resource.efficiency_rating
            })
            
        return dict(status)

    def get_resource_utilization(self, time_period: timedelta) -> Dict[str, float]:
        """Calculate resource utilization over time period."""
        end_time = datetime.now()
        start_time = end_time - time_period
        
        utilization = defaultdict(lambda: {'total_time': 0, 'used_time': 0})
        
        for resource in self.resources.values():
            util = utilization[resource.type]
            util['total_time'] += time_period.total_seconds()
            
            # Calculate used time from maintenance history
            for history in resource.maintenance_history:
                if history['start_time'] >= start_time:
                    util['used_time'] += history['duration'].total_seconds()
        
        return {
            resource_type: (util['used_time'] / util['total_time'])
            for resource_type, util in utilization.items()
        }

    async def optimize_resources(self):
        """Optimize resource allocation based on historical data."""
        # Calculate resource efficiency
        for resource in self.resources.values():
            if resource.maintenance_history:
                # Calculate average task completion time
                avg_duration = sum(
                    h['duration'].total_seconds()
                    for h in resource.maintenance_history
                ) / len(resource.maintenance_history)
                
                # Update efficiency rating
                expected_duration = self.config['standard_task_duration']
                resource.efficiency_rating = expected_duration / avg_duration
        
        # Reorder resource pools based on efficiency
        for resource_type in self.resource_pools:
            self.resource_pools[resource_type].sort(
                key=lambda r: self.resources[r].efficiency_rating,
                reverse=True
            )

# Example usage:
if __name__ == "__main__":
    # Example configuration
    config = {
        'resources': [
            {
                'id': 'TECH_001',
                'type': 'technician',
                'capabilities': ['mechanical', 'electrical'],
                'schedule': {
                    '2024-03-16': [
                        (datetime(2024, 3, 16, 8), datetime(2024, 3, 16, 16))
                    ]
                },
                'cost_per_hour': 50.0
            },
            {
                'id': 'TOOL_001',
                'type': 'tool',
                'capabilities': ['precision_measurement'],
                'cost_per_hour': 25.0
            }
        ],
        'standard_task_duration': 3600  # 1 hour in seconds
    }
    
    # Create resource manager
    resource_manager = ResourceManager(config)
    
    # Example resource request
    async def example_usage():
        request = ResourceRequest(
            request_id="REQ_001",
            task_id="TASK_001",
            resource_type="technician",
            quantity=1,
            priority=1.0,
            start_time=datetime.now(),
            duration=timedelta(hours=2),
            capabilities_required=['mechanical']
        )
        
        # Request resources
        success, assigned_resources = await resource_manager.request_resources(request)
        print(f"Resource request success: {success}")
        print(f"Assigned resources: {assigned_resources}")
        
        # Get resource status
        status = resource_manager.get_resource_status()
        print(f"Resource status: {status}")
        
        # Calculate utilization
        utilization = resource_manager.get_resource_utilization(timedelta(days=1))
        print(f"Resource utilization: {utilization}")
    
    # Run example
    asyncio.run(example_usage())