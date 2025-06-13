"""
Universal Astroquery Wrapper for Astro MCP

Provides automatic discovery and access to all astroquery services
without manual integration of each service.
"""

import importlib
import pkgutil
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import astroquery
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from .base import BaseDataSource
from .astroquery_metadata import get_service_info, ASTROQUERY_SERVICE_INFO

logger = logging.getLogger(__name__)


class AstroqueryUniversal(BaseDataSource):
    """Universal wrapper for all astroquery services."""
    
    def __init__(self, base_dir: str = None):
        super().__init__(base_dir=base_dir, source_name="astroquery")
        self._services = {}
        self._service_metadata = {}
        self._discover_services()
    
    def _discover_services(self):
        """Automatically discover all available astroquery services."""
        logger.info("Discovering astroquery services...")
        
        for importer, modname, ispkg in pkgutil.iter_modules(astroquery.__path__):
            if ispkg and modname not in ['utils', 'extern', 'solarsystem']:
                try:
                    # Import the module
                    module = importlib.import_module(f'astroquery.{modname}')
                    
                    # Find the main query class
                    service_class = self._find_service_class(module, modname)
                    
                    if service_class:
                        # Get enhanced metadata from our metadata system
                        enhanced_metadata = get_service_info(modname)
                        
                        self._services[modname] = {
                            'module': module,
                            'class': service_class,
                            'instance': None,  # Lazy instantiation
                            'capabilities': self._detect_capabilities(service_class),
                            'description': enhanced_metadata.get('description', self._extract_description(module, service_class)),
                            'requires_auth': enhanced_metadata.get('requires_auth', self._check_authentication(service_class)),
                            'full_name': enhanced_metadata.get('full_name', f'{modname.upper()} Service'),
                            'data_types': enhanced_metadata.get('data_types', ['unknown']),
                            'wavelength_coverage': enhanced_metadata.get('wavelength_coverage', 'unknown'),
                            'object_types': enhanced_metadata.get('object_types', 'unknown'),
                            'example_queries': enhanced_metadata.get('example_queries', [])
                        }
                        logger.info(f"Discovered service: {modname}")
                except Exception as e:
                    logger.warning(f"Could not load service {modname}: {e}")
        
        logger.info(f"Discovered {len(self._services)} astroquery services")
    
    def _find_service_class(self, module, modname):
        """Find the main query class in a module."""
        # Common patterns for main class names
        potential_names = [
            modname.capitalize(),
            modname.upper(),
            f"{modname.capitalize()}Class",
            modname.replace('_', '').capitalize()
        ]
        
        for name in potential_names:
            if hasattr(module, name):
                cls = getattr(module, name)
                if isinstance(cls, type):
                    return cls
        
        # Fallback: look for a class with query methods
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and hasattr(obj, 'query_region'):
                return obj
        
        return None
    
    def _detect_capabilities(self, service_class):
        """Detect what query methods a service supports."""
        capabilities = {}
        
        # Check for standard query methods
        standard_methods = [
            'query_object', 'query_region', 'query_criteria',
            'get_images', 'get_image_list', 'query', 'query_async'
        ]
        
        for method in standard_methods:
            if hasattr(service_class, method):
                capabilities[method] = True
        
        # Find all query_* methods
        for attr in dir(service_class):
            if attr.startswith('query_') and callable(getattr(service_class, attr, None)):
                capabilities[attr] = True
        
        return capabilities
    
    def _extract_description(self, module, service_class):
        """Extract a description from the module or class docstring."""
        if service_class.__doc__:
            return inspect.cleandoc(service_class.__doc__).split('\\n')[0]
        if module.__doc__:
            return inspect.cleandoc(module.__doc__).split('\\n')[0]
        return "No description available."

    def _check_authentication(self, service_class):
        """Check if the service likely requires authentication."""
        # Heuristic: check for methods like 'login' or '_login'
        for attr in dir(service_class):
            if attr.lower() in ['login', '_login']:
                return True
        return False

    def list_services(self) -> List[Dict[str, Any]]:
        """Return a list of discovered services with their enhanced metadata."""
        service_list = []
        for name, meta in self._services.items():
            service_list.append({
                "name": name,
                "full_name": meta['full_name'],
                "description": meta['description'],
                "data_types": meta['data_types'],
                "wavelength_coverage": meta['wavelength_coverage'],
                "object_types": meta['object_types'],
                "capabilities": list(meta['capabilities'].keys()),
                "requires_auth": meta['requires_auth'],
                "example_queries": meta['example_queries']
            })
        return sorted(service_list, key=lambda x: x['name'])

    def get_service_details(self, service_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific service."""
        if service_name not in self._services:
            raise ValueError(f"Unknown service: {service_name}")
        
        meta = self._services[service_name]
        return {
            "name": service_name,
            "full_name": meta['full_name'],
            "description": meta['description'],
            "data_types": meta['data_types'],
            "wavelength_coverage": meta['wavelength_coverage'],
            "object_types": meta['object_types'],
            "capabilities": list(meta['capabilities'].keys()),
            "requires_auth": meta['requires_auth'],
            "example_queries": meta['example_queries'],
            "module_path": f"astroquery.{service_name}",
            "class_name": meta['class'].__name__ if meta['class'] else "Unknown"
        }

    def search_services(self, **criteria) -> List[str]:
        """Search for services based on various criteria."""
        matching_services = []
        
        for service_name, meta in self._services.items():
            match = True
            
            # Filter by data type
            if 'data_type' in criteria:
                data_type = criteria['data_type'].lower()
                service_data_types = [dt.lower() for dt in meta['data_types']]
                if data_type not in service_data_types:
                    match = False
            
            # Filter by wavelength coverage
            if 'wavelength' in criteria:
                wavelength = criteria['wavelength'].lower()
                coverage = meta['wavelength_coverage'].lower()
                if wavelength not in coverage and coverage != 'all':
                    match = False
            
            # Filter by object type
            if 'object_type' in criteria:
                object_type = criteria['object_type'].lower()
                if isinstance(meta['object_types'], list):
                    service_object_types = [ot.lower() for ot in meta['object_types']]
                    if object_type not in service_object_types and 'all' not in service_object_types:
                        match = False
                elif meta['object_types'] != 'all' and object_type not in meta['object_types'].lower():
                    match = False
            
            # Filter by capability
            if 'capability' in criteria:
                capability = criteria['capability']
                if capability not in meta['capabilities']:
                    match = False
            
            # Filter by authentication requirement
            if 'requires_auth' in criteria:
                if meta['requires_auth'] != criteria['requires_auth']:
                    match = False
            
            if match:
                matching_services.append(service_name)
        
        return sorted(matching_services)

    def get_service(self, service_name: str):
        """Get or create a service instance."""
        if service_name not in self._services:
            raise ValueError(f"Unknown service: {service_name}")
        
        service_info = self._services[service_name]
        
        # Lazy instantiation
        if service_info['instance'] is None:
            try:
                service_info['instance'] = service_info['class']()
            except:
                # Some services might need different instantiation
                service_info['instance'] = service_info['class']
        
        return service_info['instance'] 