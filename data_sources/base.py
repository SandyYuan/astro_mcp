"""
Base Data Source Class

Provides common functionality for all astronomical data sources.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


class BaseDataSource:
    """
    Base class for astronomical data sources with unified file management.
    
    Provides common functionality for:
    - File saving and organization
    - Registry management
    - Metadata tracking
    - Error handling
    """
    
    def __init__(self, base_dir: str = None, source_name: str = "unknown"):
        """
        Initialize base data source with file management.
        
        Args:
            base_dir: Base directory for file storage
            source_name: Name of the data source (e.g., "desi", "act")
        """
        self.source_name = source_name
        
        # Set up file management
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            # Use environment variable or default to user's home directory
            import os
            env_dir = os.environ.get('ASTRO_MCP_DATA_DIR')
            if env_dir:
                self.base_dir = Path(env_dir)
            else:
                self.base_dir = Path.home() / 'astro_mcp_data'
        
        self.base_dir = self.base_dir.expanduser().resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file registry
        self._load_registry()
    
    def _load_registry(self):
        """Load existing file registry or create new one."""
        registry_path = self.base_dir / 'file_registry.json'
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                'files': {},
                'statistics': {
                    'total_files': 0,
                    'total_size_bytes': 0,
                    'by_type': {},
                    'by_source': {}
                }
            }
    
    def _save_registry(self):
        """Save file registry to disk."""
        registry_path = self.base_dir / 'file_registry.json'
        with open(registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def save_file(
        self,
        data: Any,
        filename: str,
        file_type: str = 'auto',
        description: str = None,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Save data with automatic organization and registry management.
        
        Args:
            data: Data to save
            filename: Base filename
            file_type: File format ('json', 'csv', 'npy', or 'auto')
            description: Human-readable description
            metadata: Additional metadata
            
        Returns:
            Dict with save result status and file information
        """
        # Sanitize filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in '._-')
        
        # Auto-detect file type if needed
        if file_type == 'auto':
            if isinstance(data, dict) or isinstance(data, list):
                file_type = 'json'
            elif isinstance(data, pd.DataFrame):
                file_type = 'csv'
            elif isinstance(data, np.ndarray):
                file_type = 'npy'
            else:
                file_type = 'json'  # Default
        
        # Ensure proper extension
        if not safe_filename.endswith(f'.{file_type}'):
            safe_filename = f"{safe_filename}.{file_type}"
        
        # Save to source-specific subdirectory
        source_dir = self.base_dir / self.source_name
        source_dir.mkdir(exist_ok=True)
        filepath = source_dir / safe_filename
        
        try:
            # Save the file
            if file_type == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            elif file_type == 'csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(filepath, index=False)
                else:
                    pd.DataFrame(data).to_csv(filepath, index=False)
            elif file_type == 'npy':
                np.save(filepath, data)
            else:
                with open(filepath, 'w') as f:
                    f.write(str(data))
            
            file_size = filepath.stat().st_size
            
            # Generate unique file ID
            file_id = hashlib.md5(f"{filepath}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            # Create file record
            file_record = {
                'id': file_id,
                'filename': str(filepath),
                'file_type': file_type,
                'source': self.source_name,
                'size_bytes': file_size,
                'created': datetime.now().isoformat(),
                'description': description or f"{self.source_name.upper()} data: {safe_filename}",
                'metadata': metadata or {}
            }
            
            # Update registry
            self.registry['files'][file_id] = file_record
            self.registry['statistics']['total_files'] += 1
            self.registry['statistics']['total_size_bytes'] += file_size
            
            # Update by_type stats
            if file_type not in self.registry['statistics']['by_type']:
                self.registry['statistics']['by_type'][file_type] = 0
            self.registry['statistics']['by_type'][file_type] += 1
            
            # Update by_source stats
            if self.source_name not in self.registry['statistics']['by_source']:
                self.registry['statistics']['by_source'][self.source_name] = 0
            self.registry['statistics']['by_source'][self.source_name] += 1
            
            self._save_registry()
            
            return {
                'status': 'success',
                'file_id': file_id,
                'filename': str(filepath),
                'file_type': file_type,
                'size_bytes': file_size,
                'created': datetime.now().isoformat(),
                'description': file_record['description']
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'filename': safe_filename
            }
    
    def get_file_info(self, identifier: str) -> Dict[str, Any]:
        """Get file metadata without loading content."""
        file_record = None
        
        # Check if identifier is a file ID
        if identifier in self.registry['files']:
            file_record = self.registry['files'][identifier]
        else:
            # Search by filename
            for fid, record in self.registry['files'].items():
                if record['filename'] == identifier or Path(record['filename']).name == identifier:
                    file_record = record
                    break
        
        if not file_record:
            return {
                'status': 'error',
                'error': f"File not found: {identifier}"
            }
        
        filepath = Path(file_record['filename'])
        if not filepath.exists():
            return {
                'status': 'error',
                'error': f"File no longer exists: {filepath}"
            }
        
        return {
            'status': 'success',
            'metadata': file_record
        }
    
    def list_files(
        self,
        file_type: str = None,
        pattern: str = None,
        sort_by: str = 'created',
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """List files with filtering and sorting."""
        # Filter by source first
        files = [f for f in self.registry['files'].values() 
                if f.get('source') == self.source_name]
        
        # Apply additional filters
        if file_type:
            files = [f for f in files if f['file_type'] == file_type]
        if pattern:
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(Path(f['filename']).name, pattern)]
        
        # Sort
        if sort_by == 'created':
            files.sort(key=lambda x: x['created'], reverse=True)
        elif sort_by == 'size':
            files.sort(key=lambda x: x['size_bytes'], reverse=True)
        elif sort_by == 'filename':
            files.sort(key=lambda x: x['filename'])
        
        # Limit
        if limit:
            files = files[:limit]
        
        return files
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file system statistics for this source."""
        # Get source-specific stats
        source_files = [f for f in self.registry['files'].values() 
                       if f.get('source') == self.source_name]
        
        stats = {
            'source': self.source_name,
            'total_files': len(source_files),
            'total_size_bytes': sum(f['size_bytes'] for f in source_files),
            'by_type': {}
        }
        
        # Calculate by_type for this source
        for f in source_files:
            ftype = f['file_type']
            if ftype not in stats['by_type']:
                stats['by_type'][ftype] = 0
            stats['by_type'][ftype] += 1
        
        # Add recent files
        recent_files = sorted(source_files, key=lambda x: x['created'], reverse=True)[:5]
        stats['recent_files'] = [
            {'filename': Path(f['filename']).name, 'created': f['created']} 
            for f in recent_files
        ]
        
        return stats 