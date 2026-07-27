"""
Utilities for Offline AI Book Reader
Helper functions and utilities for the system
"""

import sys
import os
site_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'site-packages')
if os.path.isdir(site_packages):
    sys.path.insert(0, site_packages)

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime
import psutil
import platform

from config import LOGGING_CONFIG, PERFORMANCE_CONFIG

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
    log_file = log_file or LOGGING_CONFIG["file"]
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["level"]),
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
    
    return logger

def get_system_info() -> Dict[str, Any]:
    """Get system information for optimization"""
    try:
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory information
        memory = psutil.virtual_memory()
        
        # Platform information
        platform_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        
        return {
            "cpu": {
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
                "architecture": platform.machine()
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": memory.percent
            },
            "platform": platform_info,
            "python_version": platform.python_version()
        }
        
    except Exception as e:
        logging.error(f"Error getting system info: {e}")
        return {}

def is_low_end_device() -> bool:
    """Determine if the current device is low-end"""
    try:
        system_info = get_system_info()
        
        # Check CPU cores
        cpu_count = system_info.get("cpu", {}).get("count", 0)
        if cpu_count <= 2:
            return True
        
        # Check memory
        memory_gb = system_info.get("memory", {}).get("total_gb", 0)
        if memory_gb <= 4:
            return True
        
        # Check if it's a mobile/embedded platform
        platform_system = system_info.get("platform", {}).get("system", "").lower()
        if platform_system in ["android", "ios", "linux"]:
            return True
        
        return False
        
    except Exception as e:
        logging.error(f"Error determining device type: {e}")
        return True  # Assume low-end if we can't determine

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def create_progress_bar(total: int, description: str = "") -> Any:
    """Create a progress bar for long operations"""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=description, unit="items")
    except ImportError:
        # Fallback to simple progress indicator
        class SimpleProgress:
            def __init__(self, total, description=""):
                self.total = total
                self.current = 0
                self.description = description
                print(f"{description}: 0/{total}")
            
            def update(self, n=1):
                self.current += n
                if self.current % max(1, self.total // 10) == 0:
                    print(f"{self.description}: {self.current}/{self.total}")
            
            def close(self):
                print(f"{self.description}: {self.current}/{self.total} - Complete")
        
        return SimpleProgress(total, description)

def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """Save data to JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        logging.error(f"Error saving JSON to {file_path}: {e}")
        return False

def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON from {file_path}: {e}")
        return None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename

def get_file_hash(file_path: str) -> str:
    """Get SHA-256 hash of a file"""
    import hashlib
    
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logging.error(f"Error calculating file hash: {e}")
        return ""

def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        if not PERFORMANCE_CONFIG["enable_monitoring"]:
            return func(*args, **kwargs)
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logging.info(f"Performance: {func.__name__} took {format_time(execution_time)}, "
                        f"memory delta: {memory_delta:+.1f}MB")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logging.error(f"Performance: {func.__name__} failed after {format_time(execution_time)}: {e}")
            raise
    
    return wrapper

def cleanup_temp_files(temp_dir: str, max_age_hours: int = 24) -> int:
    """Clean up temporary files older than specified age"""
    try:
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            return 0
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0
        
        for file_path in temp_path.rglob("*"):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        logging.warning(f"Could not delete temp file {file_path}: {e}")
        
        if deleted_count > 0:
            logging.info(f"Cleaned up {deleted_count} temporary files")
        
        return deleted_count
        
    except Exception as e:
        logging.error(f"Error cleaning up temp files: {e}")
        return 0

def validate_file_path(file_path: str) -> bool:
    """Validate if file path is safe and accessible"""
    try:
        path = Path(file_path)
        
        # Check if path exists
        if not path.exists():
            return False
        
        # Check if it's a file
        if not path.is_file():
            return False
        
        # Check if file is readable
        if not os.access(path, os.R_OK):
            return False
        
        # Check file size (prevent processing extremely large files)
        file_size = path.stat().st_size
        max_size = 500 * 1024 * 1024  # 500MB limit
        if file_size > max_size:
            logging.warning(f"File {file_path} is too large ({format_file_size(file_size)})")
            return False
        
        return True
        
    except Exception as e:
        logging.error(f"Error validating file path {file_path}: {e}")
        return False

def get_available_disk_space(path: str) -> float:
    """Get available disk space in GB"""
    try:
        disk_usage = psutil.disk_usage(path)
        return disk_usage.free / (1024**3)  # Convert to GB
    except Exception as e:
        logging.error(f"Error getting disk space: {e}")
        return 0.0

def check_disk_space(path: str, required_gb: float) -> bool:
    """Check if there's enough disk space"""
    available_gb = get_available_disk_space(path)
    return available_gb >= required_gb

def create_backup(file_path: str, backup_dir: str) -> Optional[str]:
    """Create a backup of a file"""
    try:
        source_path = Path(file_path)
        if not source_path.exists():
            return None
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
        backup_file_path = backup_path / backup_filename
        
        # Copy file
        import shutil
        shutil.copy2(source_path, backup_file_path)
        
        logging.info(f"Created backup: {backup_file_path}")
        return str(backup_file_path)
        
    except Exception as e:
        logging.error(f"Error creating backup: {e}")
        return None 