"""
Memory Manager for Offline AI Book Reader
Handles memory optimization, caching, and load-on-demand functionality
"""

import sys
import os
site_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'site-packages')
if os.path.isdir(site_packages):
    sys.path.insert(0, site_packages)

import gc
import psutil
import sqlite3
import mmap
import pickle
import gzip
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from collections import OrderedDict
import numpy as np

from config import MEMORY_CONFIG, DATABASE_CONFIG

logger = logging.getLogger(__name__)

class MemoryManager:
    """Intelligent memory management for low-end devices"""
    
    def __init__(self):
        self.max_memory_mb = MEMORY_CONFIG["max_memory_mb"]
        self.cache_size = MEMORY_CONFIG["cache_size"]
        self.compression_ratio = MEMORY_CONFIG["compression_ratio"]
        self.load_on_demand = MEMORY_CONFIG["load_on_demand"]
        self.use_memory_mapping = MEMORY_CONFIG["use_memory_mapping"]
        
        # LRU cache for embeddings
        self.embedding_cache = OrderedDict()
        self.text_cache = OrderedDict()
        
        # Memory monitoring
        self.process = psutil.Process()
        self.memory_usage_history = []
        
        logger.info(f"Memory Manager initialized with {self.max_memory_mb}MB limit")
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current memory usage in MB and percentage"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        return memory_mb, memory_percent
    
    def check_memory_limit(self) -> bool:
        """Check if current memory usage is within limits"""
        memory_mb, _ = self.get_memory_usage()
        return memory_mb < self.max_memory_mb
    
    def log_memory_usage(self, operation: str = ""):
        """Log current memory usage"""
        memory_mb, memory_percent = self.get_memory_usage()
        self.memory_usage_history.append({
            "operation": operation,
            "memory_mb": memory_mb,
            "memory_percent": memory_percent,
            "timestamp": psutil.cpu_times()
        })
        
        logger.info(f"Memory usage: {memory_mb:.1f}MB ({memory_percent:.1f}%) - {operation}")
    
    def compress_text(self, text: str) -> bytes:
        """Compress text data to save memory"""
        return gzip.compress(text.encode('utf-8'))
    
    def decompress_text(self, compressed_data: bytes) -> str:
        """Decompress text data"""
        return gzip.decompress(compressed_data).decode('utf-8')
    
    def cache_embedding(self, key: str, embedding: np.ndarray) -> None:
        """Cache an embedding with LRU eviction"""
        if len(self.embedding_cache) >= self.cache_size:
            # Remove least recently used item
            self.embedding_cache.popitem(last=False)
        
        self.embedding_cache[key] = embedding
        self.embedding_cache.move_to_end(key)
        
        # Check memory limits
        if not self.check_memory_limit():
            self.cleanup_cache()
    
    def get_cached_embedding(self, key: str) -> Optional[np.ndarray]:
        """Get cached embedding if available"""
        if key in self.embedding_cache:
            # Move to end (most recently used)
            self.embedding_cache.move_to_end(key)
            return self.embedding_cache[key]
        return None
    
    def cache_text(self, key: str, text: str) -> None:
        """Cache compressed text with LRU eviction"""
        if len(self.text_cache) >= self.cache_size:
            self.text_cache.popitem(last=False)
        
        # Compress text to save memory
        compressed_text = self.compress_text(text)
        self.text_cache[key] = compressed_text
        self.text_cache.move_to_end(key)
    
    def get_cached_text(self, key: str) -> Optional[str]:
        """Get cached text if available"""
        if key in self.text_cache:
            self.text_cache.move_to_end(key)
            compressed_text = self.text_cache[key]
            return self.decompress_text(compressed_text)
        return None
    
    def cleanup_cache(self) -> None:
        """Clean up cache when memory limit is reached"""
        logger.info("Cleaning up cache due to memory pressure")
        
        # Remove 20% of cache items
        items_to_remove = len(self.embedding_cache) // 5
        for _ in range(items_to_remove):
            if self.embedding_cache:
                self.embedding_cache.popitem(last=False)
        
        items_to_remove = len(self.text_cache) // 5
        for _ in range(items_to_remove):
            if self.text_cache:
                self.text_cache.popitem(last=False)
        
        # Force garbage collection
        gc.collect()
        
        logger.info(f"Cache cleanup complete. Embeddings: {len(self.embedding_cache)}, Texts: {len(self.text_cache)}")
    
    def create_memory_mapped_connection(self, db_path: str) -> sqlite3.Connection:
        """Create SQLite connection with memory mapping"""
        conn = sqlite3.connect(db_path)
        
        # Enable memory mapping
        if self.use_memory_mapping:
            conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
            conn.execute("PRAGMA cache_size = 10000")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        
        return conn
    
    def load_vectors_on_demand(self, book_id: str, page_numbers: List[int]) -> Dict[int, np.ndarray]:
        """Load vectors only for specific pages (load-on-demand)"""
        vectors = {}
        
        for page_num in page_numbers:
            cache_key = f"{book_id}_page_{page_num}"
            
            # Check cache first
            cached_vector = self.get_cached_embedding(cache_key)
            if cached_vector is not None:
                vectors[page_num] = cached_vector
                continue
            
            # Load from database if not in cache
            try:
                vector = self._load_vector_from_db(book_id, page_num)
                if vector is not None:
                    self.cache_embedding(cache_key, vector)
                    vectors[page_num] = vector
            except Exception as e:
                logger.error(f"Error loading vector for {book_id} page {page_num}: {e}")
        
        return vectors
    
    def _load_vector_from_db(self, book_id: str, page_num: int) -> Optional[np.ndarray]:
        """Load a single vector from database"""
        # This would be implemented in vector_db.py
        # For now, return None as placeholder
        return None
    
    def store_compressed_text(self, book_id: str, page_num: int, text: str) -> None:
        """Store compressed text in database"""
        compressed_text = self.compress_text(text)
        cache_key = f"{book_id}_text_{page_num}"
        self.cache_text(cache_key, text)
        
        # Store in database (implemented in vector_db.py)
        pass
    
    def get_compressed_text(self, book_id: str, page_num: int) -> Optional[str]:
        """Get compressed text from cache or database"""
        cache_key = f"{book_id}_text_{page_num}"
        
        # Check cache first
        cached_text = self.get_cached_text(cache_key)
        if cached_text is not None:
            return cached_text
        
        # Load from database if not in cache
        try:
            text = self._load_text_from_db(book_id, page_num)
            if text is not None:
                self.cache_text(cache_key, text)
            return text
        except Exception as e:
            logger.error(f"Error loading text for {book_id} page {page_num}: {e}")
            return None
    
    def _load_text_from_db(self, book_id: str, page_num: int) -> Optional[str]:
        """Load compressed text from database"""
        # This would be implemented in vector_db.py
        # For now, return None as placeholder
        return None
    
    def optimize_for_low_end_device(self) -> None:
        """Apply optimizations for low-end devices"""
        logger.info("Applying low-end device optimizations")
        
        # Reduce cache sizes
        self.cache_size = min(self.cache_size, 500)
        
        # Clear existing caches
        self.embedding_cache.clear()
        self.text_cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Set lower memory limit
        self.max_memory_mb = min(self.max_memory_mb, 256)
        
        logger.info(f"Optimized for low-end device: {self.max_memory_mb}MB limit, {self.cache_size} cache size")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        memory_mb, memory_percent = self.get_memory_usage()
        
        return {
            "current_memory_mb": memory_mb,
            "memory_percent": memory_percent,
            "max_memory_mb": self.max_memory_mb,
            "embedding_cache_size": len(self.embedding_cache),
            "text_cache_size": len(self.text_cache),
            "cache_size_limit": self.cache_size,
            "memory_usage_history": self.memory_usage_history[-10:],  # Last 10 entries
        }
    
    def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up Memory Manager")
        
        # Clear caches
        self.embedding_cache.clear()
        self.text_cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Log final memory usage
        self.log_memory_usage("cleanup") 