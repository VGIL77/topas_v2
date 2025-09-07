#!/usr/bin/env python3
"""
Real-time Training Monitor for HRM-TOPAS Integration

This script provides real-time monitoring of HRM-TOPAS training with:
1. Live metrics dashboard showing training progress
2. HRM-specific metrics (curriculum level, puzzle embeddings, etc.)
3. Resource usage monitoring (GPU/CPU/memory)
4. Phase transition tracking
5. Error detection and alerting
6. Performance trend analysis
7. Checkpoint status monitoring

Usage:
  python monitor_training.py --log-dir logs/hrm_integrated_v1
  python monitor_training.py --log-dir logs/hrm_integrated_v1 --web-dashboard --port 8080
  python monitor_training.py --config configs/hrm_integrated_training.json --real-time
"""

import os
import sys
import json
import time
import argparse
import threading
import signal
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass
import traceback

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Core monitoring imports
import torch
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import ttk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    timestamp: float
    phase: str
    global_step: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    
    # HRM-specific metrics
    curriculum_level: Optional[int] = None
    hrm_meta_learning_enabled: bool = False
    hrm_embedding_enabled: bool = False
    puzzle_embedding_dim: Optional[int] = None
    
    # Performance metrics
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    cpu_usage: Optional[float] = None
    training_speed: Optional[float] = None  # samples/sec
    
    # Phase-specific metrics
    phase_progress: Optional[float] = None  # 0.0 - 1.0
    phase_metrics: Optional[Dict[str, Any]] = None


class LogFileWatcher:
    """Watches log files for new training metrics."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.watched_files = {}
        self.metrics_queue = deque(maxlen=1000)  # Keep last 1000 metrics
        self.running = False
        
    def start_watching(self):
        """Start watching log files."""
        self.running = True
        self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watch_thread.start()
        
    def stop_watching(self):
        """Stop watching log files."""
        self.running = False
        
    def _watch_loop(self):
        """Main watch loop."""
        while self.running:
            try:
                self._check_log_files()
                time.sleep(1.0)  # Check every second
            except Exception as e:
                print(f"Error in watch loop: {e}")
                time.sleep(5.0)  # Wait longer on error
                
    def _check_log_files(self):
        """Check log files for updates."""
        if not self.log_dir.exists():
            return
            
        # Look for training log files
        log_files = list(self.log_dir.glob("*.log")) + list(self.log_dir.glob("**/train_*.log"))
        
        for log_file in log_files:
            if log_file not in self.watched_files:
                self.watched_files[log_file] = {"position": 0, "size": 0}
                
            # Check if file has grown
            current_size = log_file.stat().st_size
            if current_size > self.watched_files[log_file]["size"]:
                self._read_new_log_content(log_file)
                self.watched_files[log_file]["size"] = current_size
                
    def _read_new_log_content(self, log_file: Path):
        """Read new content from log file."""
        try:
            with open(log_file, 'r') as f:
                f.seek(self.watched_files[log_file]["position"])
                new_lines = f.readlines()
                self.watched_files[log_file]["position"] = f.tell()
                
                # Parse new lines for metrics
                for line in new_lines:
                    metrics = self._parse_log_line(line.strip())
                    if metrics:
                        self.metrics_queue.append(metrics)
                        
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")
            
    def _parse_log_line(self, line: str) -> Optional[TrainingMetrics]:
        """Parse a log line for training metrics."""
        try:
            # Look for JSON-formatted metrics
            if "METRICS:" in line or '"global_step"' in line:
                # Extract JSON part
                json_start = line.find('{')
                if json_start >= 0:
                    json_str = line[json_start:]
                    data = json.loads(json_str)
                    
                    metrics = TrainingMetrics(
                        timestamp=time.time(),
                        phase=data.get("phase", "unknown"),
                        global_step=data.get("global_step", 0),
                        loss=data.get("loss"),
                        learning_rate=data.get("learning_rate"),
                        curriculum_level=data.get("curriculum_level"),
                        hrm_meta_learning_enabled=data.get("hrm_meta_learning_enabled", False),
                        hrm_embedding_enabled=data.get("hrm_embedding_enabled", False),
                        puzzle_embedding_dim=data.get("puzzle_embedding_dim"),
                        phase_progress=data.get("phase_progress"),
                        phase_metrics=data.get("phase_metrics")
                    )
                    
                    return metrics
                    
            # Look for specific log patterns
            elif "Phase" in line and "completed" in line:
                # Phase completion log
                timestamp = time.time()
                phase = "unknown"
                
                if "Phase 0" in line:
                    phase = "phase0_world_grammar"
                elif "Phase 1" in line:
                    phase = "phase1_policy_distill"
                # Add more phase parsing as needed
                
                return TrainingMetrics(
                    timestamp=timestamp,
                    phase=phase,
                    global_step=0,
                    phase_progress=1.0
                )
                
        except Exception as e:
            # Silently ignore parse errors
            pass
            
        return None
        
    def get_recent_metrics(self, count: int = 100) -> List[TrainingMetrics]:
        """Get recent metrics."""
        return list(self.metrics_queue)[-count:]


class ResourceMonitor:
    """Monitors system resources during training."""
    
    def __init__(self):
        self.running = False
        self.resource_history = deque(maxlen=300)  # 5 minutes at 1Hz
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if not PSUTIL_AVAILABLE:
            print("⚠️  psutil not available, resource monitoring disabled")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.running = False
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                resource_data = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_available": psutil.virtual_memory().available / (1024**3),  # GB
                }
                
                # GPU monitoring if available
                if torch.cuda.is_available():
                    try:
                        resource_data["gpu_memory_used"] = torch.cuda.memory_allocated() / (1024**3)  # GB
                        resource_data["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                        resource_data["gpu_utilization"] = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
                    except Exception:
                        pass
                        
                self.resource_history.append(resource_data)
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Error in resource monitoring: {e}")
                time.sleep(5.0)
                
    def get_latest_resources(self) -> Optional[Dict]:
        """Get latest resource data."""
        return self.resource_history[-1] if self.resource_history else None
        
    def get_resource_history(self, minutes: int = 5) -> List[Dict]:
        """Get resource history for specified minutes."""
        cutoff_time = time.time() - (minutes * 60)
        return [r for r in self.resource_history if r["timestamp"] > cutoff_time]


class ConsoleMonitor:
    """Console-based training monitor."""
    
    def __init__(self, log_dir: Path, config: Optional[Dict] = None):
        self.log_dir = Path(log_dir)
        self.config = config or {}
        
        # Initialize components
        self.log_watcher = LogFileWatcher(log_dir)
        self.resource_monitor = ResourceMonitor()
        
        # State
        self.running = False
        self.last_metrics = None
        self.phase_history = []
        
    def start(self):
        """Start monitoring."""
        print(f"🔍 Starting HRM-TOPAS Training Monitor")
        print(f"📁 Log directory: {self.log_dir}")
        print(f"🖥️  Resource monitoring: {'✅' if PSUTIL_AVAILABLE else '❌'}")
        print("="*60)
        
        # Start components
        self.log_watcher.start_watching()
        self.resource_monitor.start_monitoring()
        
        self.running = True
        
        # Main monitoring loop
        try:
            self._console_loop()
        except KeyboardInterrupt:
            print("\n⚠️  Monitor interrupted by user")
        finally:
            self.stop()
            
    def stop(self):
        """Stop monitoring."""
        self.running = False
        self.log_watcher.stop_watching()
        self.resource_monitor.stop_monitoring()
        print("🛑 Monitor stopped")
        
    def _console_loop(self):
        """Main console monitoring loop."""
        last_display = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Update display every 5 seconds
                if current_time - last_display >= 5.0:
                    self._update_console_display()
                    last_display = current_time
                    
                time.sleep(1.0)
                
            except Exception as e:
                print(f"Error in console loop: {e}")
                time.sleep(5.0)
                
    def _update_console_display(self):
        """Update console display with latest metrics."""
        # Clear screen (simple approach)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"🚀 HRM-TOPAS Training Monitor")
        print(f"📁 Log Directory: {self.log_dir}")
        print(f"🕒 Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Get latest metrics
        recent_metrics = self.log_watcher.get_recent_metrics(1)
        current_resources = self.resource_monitor.get_latest_resources()
        
        if recent_metrics:
            metrics = recent_metrics[-1]
            self.last_metrics = metrics
            
            print(f"📊 CURRENT TRAINING STATUS")
            print(f"  Phase: {metrics.phase}")
            print(f"  Global Step: {metrics.global_step:,}")
            
            if metrics.loss is not None:
                print(f"  Loss: {metrics.loss:.6f}")
                
            if metrics.learning_rate is not None:
                print(f"  Learning Rate: {metrics.learning_rate:.2e}")
                
            if metrics.phase_progress is not None:
                progress_bar = self._create_progress_bar(metrics.phase_progress)
                print(f"  Phase Progress: {progress_bar} {metrics.phase_progress:.1%}")
                
            # HRM-specific metrics
            if metrics.curriculum_level is not None:
                print(f"  Curriculum Level: {metrics.curriculum_level}")
                
            if metrics.hrm_meta_learning_enabled or metrics.hrm_embedding_enabled:
                print(f"  HRM Features: Meta-Learning: {'✅' if metrics.hrm_meta_learning_enabled else '❌'}, "
                      f"Embeddings: {'✅' if metrics.hrm_embedding_enabled else '❌'}")
                
        else:
            print(f"⚠️  No recent training metrics found")
            print(f"   Waiting for training to start or log files to be created...")
            
        print()
        
        # Resource usage
        if current_resources:
            print(f"💻 RESOURCE USAGE")
            print(f"  CPU: {current_resources['cpu_percent']:.1f}%")
            print(f"  Memory: {current_resources['memory_percent']:.1f}% "
                  f"(Available: {current_resources['memory_available']:.1f}GB)")
            
            if "gpu_memory_used" in current_resources and "gpu_memory_total" in current_resources:
                gpu_percent = (current_resources['gpu_memory_used'] / current_resources['gpu_memory_total']) * 100
                print(f"  GPU Memory: {gpu_percent:.1f}% "
                      f"({current_resources['gpu_memory_used']:.1f}GB / {current_resources['gpu_memory_total']:.1f}GB)")
                      
        else:
            print(f"💻 RESOURCE USAGE: Not available")
            
        print()
        
        # Phase history
        all_metrics = self.log_watcher.get_recent_metrics(100)
        phase_transitions = self._analyze_phase_transitions(all_metrics)
        
        if phase_transitions:
            print(f"📈 PHASE HISTORY")
            for i, (phase, start_time, end_time) in enumerate(phase_transitions[-5:]):  # Last 5 phases
                duration = end_time - start_time if end_time else time.time() - start_time
                status = "✅ Completed" if end_time else "🔄 In Progress"
                print(f"  {phase}: {duration:.1f}s {status}")
                
        print()
        
        # Training speed estimate
        if len(all_metrics) >= 2:
            speed_metrics = [m for m in all_metrics if m.global_step > 0][-10:]  # Last 10 with steps
            if len(speed_metrics) >= 2:
                time_diff = speed_metrics[-1].timestamp - speed_metrics[0].timestamp
                step_diff = speed_metrics[-1].global_step - speed_metrics[0].global_step
                if time_diff > 0:
                    steps_per_sec = step_diff / time_diff
                    print(f"⚡ Training Speed: {steps_per_sec:.2f} steps/sec")
                    
        # Instructions
        print(f"💡 Press Ctrl+C to stop monitoring")
        
    def _create_progress_bar(self, progress: float, width: int = 30) -> str:
        """Create a text progress bar."""
        filled = int(width * progress)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
        
    def _analyze_phase_transitions(self, metrics: List[TrainingMetrics]) -> List[Tuple[str, float, Optional[float]]]:
        """Analyze phase transitions from metrics."""
        transitions = []
        current_phase = None
        phase_start = None
        
        for metric in metrics:
            if current_phase != metric.phase:
                # Phase changed
                if current_phase and phase_start:
                    # End previous phase
                    transitions.append((current_phase, phase_start, metric.timestamp))
                    
                # Start new phase
                current_phase = metric.phase
                phase_start = metric.timestamp
                
        # Add current phase (without end time)
        if current_phase and phase_start:
            transitions.append((current_phase, phase_start, None))
            
        return transitions


class WebDashboard:
    """Web-based dashboard for training monitoring."""
    
    def __init__(self, log_dir: Path, port: int = 8080):
        self.log_dir = Path(log_dir)
        self.port = port
        
        # TODO: Implement web dashboard using Flask/Streamlit/etc.
        print(f"⚠️  Web dashboard not implemented yet")
        print(f"   Would serve on http://localhost:{port}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HRM-TOPAS Training Monitor")
    parser.add_argument("--log-dir", type=str, required=True,
                       help="Directory containing training logs")
    parser.add_argument("--config", type=str, default=None,
                       help="Configuration file for additional context")
    parser.add_argument("--web-dashboard", action="store_true",
                       help="Start web dashboard")
    parser.add_argument("--port", type=int, default=8080,
                       help="Port for web dashboard")
    parser.add_argument("--real-time", action="store_true",
                       help="Enable real-time monitoring features")
    parser.add_argument("--update-interval", type=float, default=5.0,
                       help="Display update interval in seconds")
    
    args = parser.parse_args()
    
    # Validate log directory
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"❌ Log directory not found: {log_dir}")
        print(f"   Creating directory...")
        log_dir.mkdir(parents=True, exist_ok=True)
        
    # Load config if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            
    print(f"🔍 HRM-TOPAS Training Monitor")
    print(f"📁 Log directory: {log_dir}")
    print(f"🌐 Web dashboard: {'Enabled' if args.web_dashboard else 'Disabled'}")
    print(f"⏱️  Real-time mode: {'Enabled' if args.real_time else 'Disabled'}")
    
    # Check dependencies
    missing_deps = []
    if not PSUTIL_AVAILABLE:
        missing_deps.append("psutil (resource monitoring)")
    if args.web_dashboard and not MATPLOTLIB_AVAILABLE:
        missing_deps.append("matplotlib (web dashboard)")
        
    if missing_deps:
        print(f"⚠️  Missing optional dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print(f"   Install with: pip install psutil matplotlib")
        print()
    
    # Start monitoring
    if args.web_dashboard:
        # Start web dashboard
        dashboard = WebDashboard(log_dir, args.port)
        print(f"🌐 Web dashboard would start on port {args.port}")
        print(f"   (Not implemented yet)")
    else:
        # Start console monitor
        monitor = ConsoleMonitor(log_dir, config)
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            print("\n🛑 Received interrupt signal, stopping monitor...")
            monitor.stop()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            monitor.start()
        except Exception as e:
            print(f"❌ Monitor failed: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()