"""
Dynamic Hardware-Aware Thread Scaling — monitors I/O vs compute phases
in the Jafar learning loop and adjusts llama-server process priority
and thread hints to optimise resource utilisation on CPU-only Windows.

Step classification:
    HEAVY_IO      — book processing, deep understanding (disk-bound)
    LIGHT_IO      — pattern recognition (mixed)
    COMPUTE       — reasoning, planning, reflection (CPU-bound, elastic)
    HEAVY_COMPUTE — truth system, contradiction resolution, world model
    LLM           — transformer reasoning (maximise llama-server throughput)
    LIGHT         — goals, curiosity, agentic (low resource)
"""

import logging
import os
import time
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

# Priority mapping for Windows
PRIORITY_CLASSES = {
    "idle": 0x00000040,           # IDLE_PRIORITY_CLASS
    "below_normal": 0x00004000,   # BELOW_NORMAL_PRIORITY_CLASS
    "normal": 0x00000020,         # NORMAL_PRIORITY_CLASS
    "above_normal": 0x00008000,   # ABOVE_NORMAL_PRIORITY_CLASS
    "high": 0x00000080,           # HIGH_PRIORITY_CLASS
    "realtime": 0x00000100,       # REALTIME_PRIORITY_CLASS
}

# Step types mapped to resource profiles
STEP_PROFILES = {
    "heuristic_io": ("below_normal", "idle"),
    "light_io": ("normal", "below_normal"),
    "compute": ("normal", "normal"),
    "heavy_compute": ("above_normal", "above_normal"),
    "llm": ("normal", "high"),
    "light": ("below_normal", "below_normal"),
}

# Learning-loop step name -> profile
STEP_CLASSIFICATION = {
    "0_root_cause_analysis": "light_io",
    "1_process_books": "heuristic_io",
    "1b_deep_understanding": "heuristic_io",
    "2_pattern_recognition": "light_io",
    "3_reasoning": "compute",
    "3b_transformer_reasoning": "llm",
    "3c_planning_reasoning": "compute",
    "4_reflection": "compute",
    "5_curiosity": "light",
    "6_truth_decay": "light_io",
    "6b_truth_system": "heavy_compute",
    "6c_contradiction_resolution": "heavy_compute",
    "6d_activation_spread": "heavy_compute",
    "6e_highway_prediction": "heavy_compute",
    "7_goals": "light",
    "8_world_model": "heavy_compute",
    "9_agentic_learning": "light",
}


class HardwareMonitor:
    """Monitors system resources and adjusts process priorities per step type.

    On Windows, uses psutil and Win32 process priority classes.
    Gracefully degrades when psutil is unavailable.
    """

    def __init__(self, llama_server_process: Optional[Any] = None,
                 enable_scaling: bool = True):
        self._llama_proc = llama_server_process
        self._enable_scaling = enable_scaling
        self._last_profile: Optional[str] = None
        self._last_adjust_time = 0.0
        self._adjustment_count = 0
        self._min_adjust_interval = 2.0  # seconds between adjustments

        self._main_pid = os.getpid()
        self._main_proc: Optional[Any] = None
        if HAS_PSUTIL:
            try:
                self._main_proc = psutil.Process(self._main_pid)
            except Exception:
                self._main_proc = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def set_llama_server_process(self, process: Any) -> None:
        """Set the llama-server subprocess.Popen object for priority control."""
        self._llama_proc = process

    def get_system_info(self) -> Dict[str, Any]:
        """Return current system resource snapshot."""
        info: Dict[str, Any] = {
            "enable_scaling": self._enable_scaling,
            "adjustments_made": self._adjustment_count,
            "last_profile": self._last_profile,
        }
        if HAS_PSUTIL:
            try:
                info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
                info["cpu_count_logical"] = psutil.cpu_count(logical=True)
                info["cpu_count_physical"] = psutil.cpu_count(logical=False)
                mem = psutil.virtual_memory()
                info["memory_percent"] = round(mem.percent, 1)
                info["memory_available_mb"] = round(mem.available / 1048576, 1)
            except Exception:
                pass
        return info

    def get_llama_process_info(self) -> Optional[Dict[str, Any]]:
        """Get info about the llama-server process if available."""
        if self._llama_proc is None:
            return None
        try:
            proc = psutil.Process(self._llama_proc.pid) if HAS_PSUTIL else None
            if proc is None:
                return {"pid": self._llama_proc.pid, "psutil": False}
            return {
                "pid": self._llama_proc.pid,
                "cpu_percent": proc.cpu_percent(interval=0.1),
                "memory_percent": round(proc.memory_percent(), 2),
                "memory_rss_mb": round(proc.memory_info().rss / 1048576, 1),
                "status": proc.status(),
                "nice": proc.nice(),
            }
        except Exception as e:
            return {"pid": self._llama_proc.pid, "error": str(e)}

    # ------------------------------------------------------------------
    #  Per-step adjustment
    # ------------------------------------------------------------------

    def adjust_for_step(self, step_name: str) -> Dict[str, Any]:
        """Adjust process priorities based on the upcoming step.

        Called at the beginning of each learning-loop step.
        Returns a dict describing what was adjusted.
        """
        if not self._enable_scaling or not HAS_PSUTIL:
            return {"status": "skipped", "reason": "disabled or no psutil"}

        # Rate-limit adjustments
        now = time.time()
        if now - self._last_adjust_time < self._min_adjust_interval:
            return {"status": "skipped", "reason": "rate-limited"}

        profile_name = STEP_CLASSIFICATION.get(step_name, "compute")
        profile = STEP_PROFILES.get(profile_name, STEP_PROFILES["compute"])
        main_priority, llama_priority = profile

        result: Dict[str, Any] = {
            "step": step_name,
            "profile": profile_name,
            "main_priority": main_priority,
            "llama_priority": llama_priority,
        }

        # Adjust main process priority
        main_ok = self._set_priority(self._main_proc, main_priority)
        result["main_priority_set"] = main_ok

        # Adjust llama-server priority
        llama_ok = self._set_external_priority(llama_priority)
        result["llama_priority_set"] = llama_ok

        self._last_profile = profile_name
        self._last_adjust_time = now
        self._adjustment_count += 1

        if main_ok or llama_ok:
            logger.debug("Step %s → priority profile=%s (main=%s, llama=%s)",
                         step_name, profile_name, main_priority, llama_priority)

        return result

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_priority(proc: Any, level_name: str) -> bool:
        """Set process priority class on Windows via psutil."""
        if proc is None or not HAS_PSUTIL:
            return False
        try:
            nice_map = {
                "idle": psutil.IDLE_PRIORITY_CLASS,
                "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                "normal": psutil.NORMAL_PRIORITY_CLASS,
                "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                "high": psutil.HIGH_PRIORITY_CLASS,
                "realtime": psutil.REALTIME_PRIORITY_CLASS,
            }
            nice_val = nice_map.get(level_name)
            if nice_val is not None:
                proc.nice(nice_val)
                return True
        except psutil.AccessDenied:
            debug_logger.debug("Access denied setting priority to %s", level_name)
        except Exception as e:
            debug_logger.debug("Failed to set priority %s: %s", level_name, e)
        return False

    def _set_external_priority(self, level_name: str) -> bool:
        """Set priority on the llama-server subprocess."""
        if self._llama_proc is None or not HAS_PSUTIL:
            return False
        try:
            proc = psutil.Process(self._llama_proc.pid)
            return self._set_priority(proc, level_name)
        except psutil.NoSuchProcess:
            debug_logger.debug("llama-server process gone, skipping priority")
        except Exception as e:
            debug_logger.debug("Failed to set llama priority: %s", e)
        return False
