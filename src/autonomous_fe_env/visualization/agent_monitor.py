"""
Agent monitor for tracking VULCAN agent activities.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentMonitor:
    """
    Monitor for tracking agent activities and performance.

    Provides real-time monitoring of agent activities, performance metrics,
    and communication patterns in the VULCAN system.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.start_time = time.time()

        # Agent tracking data
        self.agent_activities = defaultdict(list)
        self.agent_performance = defaultdict(dict)
        self.agent_communications = []
        self.agent_status = {}

        # Real-time monitoring
        self.activity_buffer = deque(maxlen=100)
        self.performance_buffer = deque(maxlen=50)

        # Monitoring settings
        self.monitor_config = config.get("monitoring", {})
        self.log_level = self.monitor_config.get("log_level", "INFO")
        self.buffer_size = self.monitor_config.get("buffer_size", 100)

        # Thread safety
        self.lock = threading.Lock()

    def log_agent_activity(
        self,
        agent_name: str,
        activity_type: str,
        details: Dict[str, Any],
        duration: Optional[float] = None,
    ) -> None:
        """
        Log an agent activity.

        Args:
            agent_name: Name of the agent
            activity_type: Type of activity (e.g., 'feature_generation', 'reflection')
            details: Additional details about the activity
            duration: Optional duration of the activity
        """
        timestamp = time.time()

        activity_record = {
            "timestamp": timestamp,
            "agent": agent_name,
            "activity_type": activity_type,
            "details": details,
            "duration": duration,
            "relative_time": timestamp - self.start_time,
        }

        with self.lock:
            self.agent_activities[agent_name].append(activity_record)
            self.activity_buffer.append(activity_record)

            # Update agent status
            self.agent_status[agent_name] = {
                "last_activity": activity_type,
                "last_seen": timestamp,
                "status": "active",
            }

        # Log to console if configured
        if self.log_level == "DEBUG":
            logger.debug(f"Agent {agent_name}: {activity_type} - {details}")

    def log_agent_performance(
        self,
        agent_name: str,
        metric_name: str,
        value: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log agent performance metric.

        Args:
            agent_name: Name of the agent
            metric_name: Name of the performance metric
            value: Metric value
            context: Optional context information
        """
        timestamp = time.time()

        performance_record = {
            "timestamp": timestamp,
            "agent": agent_name,
            "metric": metric_name,
            "value": value,
            "context": context or {},
            "relative_time": timestamp - self.start_time,
        }

        with self.lock:
            if agent_name not in self.agent_performance:
                self.agent_performance[agent_name] = defaultdict(list)

            self.agent_performance[agent_name][metric_name].append(performance_record)
            self.performance_buffer.append(performance_record)

    def log_agent_communication(
        self, from_agent: str, to_agent: str, message_type: str, content: Dict[str, Any]
    ) -> None:
        """
        Log communication between agents.

        Args:
            from_agent: Source agent
            to_agent: Target agent
            message_type: Type of message
            content: Message content
        """
        timestamp = time.time()

        comm_record = {
            "timestamp": timestamp,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type,
            "content": content,
            "relative_time": timestamp - self.start_time,
        }

        with self.lock:
            self.agent_communications.append(comm_record)

    def get_agent_summary(self, agent_name: str) -> Dict[str, Any]:
        """
        Get summary statistics for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with agent summary
        """
        with self.lock:
            activities = self.agent_activities.get(agent_name, [])
            performance = self.agent_performance.get(agent_name, {})
            status = self.agent_status.get(agent_name, {})

        if not activities:
            return {"agent": agent_name, "status": "inactive", "activities": 0}

        # Activity statistics
        activity_types = defaultdict(int)
        total_duration = 0
        duration_count = 0

        for activity in activities:
            activity_types[activity["activity_type"]] += 1
            if activity["duration"]:
                total_duration += activity["duration"]
                duration_count += 1

        avg_duration = total_duration / duration_count if duration_count > 0 else 0

        # Performance statistics
        performance_summary = {}
        for metric_name, records in performance.items():
            values = [r["value"] for r in records]
            if values:
                performance_summary[metric_name] = {
                    "latest": values[-1],
                    "average": sum(values) / len(values),
                    "best": max(values),
                    "count": len(values),
                }

        return {
            "agent": agent_name,
            "status": status.get("status", "unknown"),
            "last_activity": status.get("last_activity", "none"),
            "last_seen": status.get("last_seen", 0),
            "total_activities": len(activities),
            "activity_breakdown": dict(activity_types),
            "average_duration": avg_duration,
            "performance_metrics": performance_summary,
            "uptime": time.time() - self.start_time if activities else 0,
        }

    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get overall system summary.

        Returns:
            Dictionary with system-wide statistics
        """
        with self.lock:
            all_agents = set(self.agent_activities.keys()) | set(
                self.agent_status.keys()
            )

            total_activities = sum(
                len(activities) for activities in self.agent_activities.values()
            )
            total_communications = len(self.agent_communications)

            # Agent status breakdown
            status_counts = defaultdict(int)
            for agent_name in all_agents:
                status = self.agent_status.get(agent_name, {}).get("status", "inactive")
                status_counts[status] += 1

            # Recent activity
            recent_activities = list(self.activity_buffer)[-10:]

            # Performance trends
            recent_performance = list(self.performance_buffer)[-20:]

        return {
            "runtime": time.time() - self.start_time,
            "total_agents": len(all_agents),
            "agent_status": dict(status_counts),
            "total_activities": total_activities,
            "total_communications": total_communications,
            "recent_activities": recent_activities,
            "recent_performance": recent_performance,
            "active_agents": [
                name
                for name, status in self.agent_status.items()
                if status.get("status") == "active"
            ],
        }

    def print_live_status(self) -> None:
        """Print live status of all agents."""
        system_summary = self.get_system_summary()

        print(f"\n{'=' * 60}")
        print(f"VULCAN AGENT MONITOR - Runtime: {system_summary['runtime']:.1f}s")
        print(f"{'=' * 60}")

        print("System Overview:")
        print(f"  Total Agents: {system_summary['total_agents']}")
        print(f"  Active Agents: {len(system_summary['active_agents'])}")
        print(f"  Total Activities: {system_summary['total_activities']}")
        print(f"  Communications: {system_summary['total_communications']}")

        # Agent status
        print("\nAgent Status:")
        for status, count in system_summary["agent_status"].items():
            print(f"  {status.title()}: {count}")

        # Active agents details
        if system_summary["active_agents"]:
            print("\nActive Agents:")
            for agent_name in system_summary["active_agents"]:
                summary = self.get_agent_summary(agent_name)
                print(f"  {agent_name}:")
                print(f"    Last Activity: {summary['last_activity']}")
                print(f"    Total Activities: {summary['total_activities']}")
                if summary["average_duration"] > 0:
                    print(f"    Avg Duration: {summary['average_duration']:.2f}s")

        # Recent activities
        if system_summary["recent_activities"]:
            print("\nRecent Activities:")
            for activity in system_summary["recent_activities"][-5:]:
                print(
                    f"  {activity['agent']}: {activity['activity_type']} "
                    f"({activity['relative_time']:.1f}s)"
                )

        print(f"{'=' * 60}\n")

    def get_agent_activity_timeline(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of agent activities.

        Args:
            agent_name: Name of the agent

        Returns:
            List of activity records sorted by timestamp
        """
        with self.lock:
            activities = self.agent_activities.get(agent_name, [])

        return sorted(activities, key=lambda x: x["timestamp"])

    def get_communication_graph(self) -> Dict[str, Any]:
        """
        Get communication graph data for visualization.

        Returns:
            Dictionary with nodes and edges for communication graph
        """
        with self.lock:
            communications = self.agent_communications.copy()

        # Build nodes and edges
        nodes = set()
        edges = defaultdict(int)

        for comm in communications:
            nodes.add(comm["from_agent"])
            nodes.add(comm["to_agent"])
            edge_key = (comm["from_agent"], comm["to_agent"])
            edges[edge_key] += 1

        return {
            "nodes": list(nodes),
            "edges": [
                {"from": edge[0], "to": edge[1], "weight": weight}
                for edge, weight in edges.items()
            ],
        }

    def export_activity_log(self, filepath: str) -> None:
        """
        Export activity log to file.

        Args:
            filepath: Path to save the log file
        """
        import json

        with self.lock:
            export_data = {
                "start_time": self.start_time,
                "export_time": time.time(),
                "agent_activities": dict(self.agent_activities),
                "agent_performance": dict(self.agent_performance),
                "agent_communications": self.agent_communications,
                "agent_status": self.agent_status,
            }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Activity log exported to {filepath}")

    def reset_monitoring(self) -> None:
        """Reset all monitoring data."""
        with self.lock:
            self.agent_activities.clear()
            self.agent_performance.clear()
            self.agent_communications.clear()
            self.agent_status.clear()
            self.activity_buffer.clear()
            self.performance_buffer.clear()
            self.start_time = time.time()

        logger.info("Agent monitoring data reset")
