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
import uuid
from pathlib import Path
import numpy as np

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
        self.source_dir = self.base_dir / self.source_name
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
                "service": name,
                "full_name": meta['full_name'],
                "description": meta['description'],
                "data_types": meta['data_types'],
                "wavelength_coverage": meta['wavelength_coverage'],
                "object_types": meta['object_types'],
                "capabilities": list(meta['capabilities'].keys()),
                "requires_auth": meta['requires_auth'],
                "example_queries": meta['example_queries']
            })
        return sorted(service_list, key=lambda x: x['service'])

    def get_service_details(self, service_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific service, including method signatures."""
        if service_name not in self._services:
            raise ValueError(f"Unknown service: {service_name}")
        
        meta = self._services[service_name]
        service_class = meta.get('class')

        # Base details
        details = {
            "service": service_name,
            "full_name": meta['full_name'],
            "description": meta['description'],
            "data_types": meta['data_types'],
            "wavelength_coverage": meta['wavelength_coverage'],
            "object_types": meta['object_types'],
            "capabilities": list(meta['capabilities'].keys()),
            "requires_auth": meta['requires_auth'],
            "example_queries": meta['example_queries'],
            "module_path": f"astroquery.{service_name}",
            "class_name": service_class.__name__ if service_class else "Unknown",
            "methods": {}
        }

        if not service_class:
            return details

        # Introspect methods to get parameters and docstrings
        for method_name in details['capabilities']:
            if hasattr(service_class, method_name):
                method = getattr(service_class, method_name)
                
                try:
                    sig = inspect.signature(method)
                    method_info = {
                        'docstring': inspect.cleandoc(method.__doc__ or "No docstring available.").split('\\n')[0],
                        'parameters': {}
                    }
                    
                    for param in sig.parameters.values():
                        # Skip self, args, kwargs
                        if param.name in ['self', 'args', 'kwargs']:
                            continue
                        
                        param_info = {}
                        if param.default is not inspect.Parameter.empty:
                            param_info['default'] = str(param.default)
                        else:
                            param_info['required'] = True
                            
                        if param.annotation is not inspect.Parameter.empty:
                            # Clean up the type annotation string
                            param_info['type'] = str(param.annotation).replace("<class '", "").replace("'>", "")
                        else:
                            param_info['type'] = 'Any'

                        method_info['parameters'][param.name] = param_info
                    
                    details['methods'][method_name] = method_info

                except (ValueError, TypeError): # Some methods may not be introspectable
                    details['methods'][method_name] = {
                        'docstring': 'Could not inspect method signature.',
                        'parameters': {}
                    }

        return details

    def search_services(self,
                        data_type: str = None,
                        wavelength: str = None,
                        object_type: str = None,
                        capability: str = None,
                        requires_auth: bool = None) -> List[Dict]:
        """Find and rank services that match specified criteria."""
        matches = []
        
        for service_name, service_info in self._services.items():
            score = 0
            match_reasons = []

            # Filter by data type
            if data_type:
                service_data_types = [dt.lower() for dt in service_info.get('data_types', [])]
                if data_type.lower() in service_data_types:
                    score += 3
                    match_reasons.append(f"provides {data_type} data")

            # Filter by wavelength coverage
            if wavelength:
                coverage = service_info.get('wavelength_coverage', '').lower()
                if wavelength.lower() in coverage or coverage == 'all':
                    score += 2
                    match_reasons.append(f"covers {wavelength} wavelengths")

            # Filter by object type
            if object_type:
                obj_types = service_info.get('object_types', 'all')
                object_type_lower = object_type.lower()
                
                match_found = False
                if isinstance(obj_types, list):
                    service_object_types = [ot.lower() for ot in obj_types]
                    if 'all' in service_object_types or object_type_lower in service_object_types:
                        match_found = True
                elif obj_types == 'all' or object_type_lower in obj_types.lower():
                    match_found = True
                
                if match_found:
                    score += 2
                    match_reasons.append(f"includes {object_type}")

            # Filter by capability
            if capability and capability in service_info['capabilities']:
                score += 3
                match_reasons.append(f"supports '{capability}'")
            
            # Filter by authentication requirement
            if requires_auth is not None and service_info['requires_auth'] == requires_auth:
                score += 1
                reason = "does not require authentication" if not requires_auth else "matches auth requirement"
                match_reasons.append(reason)
            
            if score > 0:
                matches.append({
                    'service': service_name,
                    'full_name': service_info['full_name'],
                    'score': score,
                    'reasons': match_reasons,
                    'description': service_info['description'].split('\\n')[0]
                })

        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches

    def universal_query(self, service_name: str, query_type: str = 'auto', auto_save: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Universal query interface for any astroquery service.
        
        Parameters
        ----------
        service_name : str
            Name of the astroquery service
        query_type : str
            Type of query to perform (auto-detected if 'auto')
        auto_save : bool
            Whether to automatically save results to a file (default: True)
        **kwargs : dict
            Query parameters passed to the service
        
        Returns
        -------
        dict
            Query results with status and data
        """
        try:
            # Ensure the service class is available
            if service_name not in self._services:
                raise ValueError(f"Unknown service: {service_name}")
            
            service_info = self._services[service_name]
            
            # --- AUTHENTICATION CHECK ---
            if service_info.get('requires_auth'):
                return self._generate_auth_required_help(service_name, query_type, kwargs)

            service = self.get_service(service_name)
            
            # Auto-detect query type
            if query_type == 'auto':
                query_type = self._detect_query_type(service_name, kwargs)
            
            if not hasattr(service, query_type):
                raise AttributeError(f"Service '{service_name}' does not have method '{query_type}'")

            # Parameter preprocessing
            processed_kwargs = self._preprocess_parameters(service_name, query_type, kwargs)
            
            # Execute query
            method = getattr(service, query_type)
            result = method(**processed_kwargs)
            
            # Process and save results
            return self._process_results(result, service_name, query_type, kwargs, auto_save)
            
        except Exception as e:
            logger.error(f"Query failed for {service_name}: {str(e)}")
            return {
                'status': 'error',
                'service': service_name,
                'query_type': query_type,
                'error': str(e),
                'help': self._generate_error_help(service_name, query_type, e)
            }

    def _detect_query_type(self, service_name: str, kwargs) -> str:
        """Auto-detect the appropriate query method based on parameters."""
        capabilities = self._services[service_name]['capabilities']
        
        # Check for object name query
        if any(key in kwargs for key in ['object_name', 'objectname', 'target', 'source']):
            if 'query_object' in capabilities:
                return 'query_object'
        
        # Check for coordinate query
        if 'coordinates' in kwargs or all(k in kwargs for k in ['ra', 'dec']):
            if 'query_region' in capabilities:
                return 'query_region'
        
        # Check for catalog query (Vizier specific)
        if 'catalog' in kwargs and 'query_catalog' in capabilities:
            return 'query_catalog'
        
        # Default to generic query if available
        if 'query' in capabilities:
            return 'query'
        
        # Fallback to first available high-priority query method
        for method in ['query_object', 'query_region', 'query_criteria']:
            if method in capabilities:
                return method
        
        if capabilities:
            return list(capabilities.keys())[0]

        raise ValueError(f"Could not determine appropriate query method for service {service_name}")

    def _preprocess_parameters(self, service_name: str, query_type: str, kwargs: Dict) -> Dict:
        """Preprocess parameters for compatibility."""
        processed = kwargs.copy()
        
        # Handle coordinate conversion for region/image queries
        if query_type in ['query_region', 'get_images', 'get_image_list']:
            if 'ra' in processed and 'dec' in processed and 'coordinates' not in processed:
                from astropy.coordinates import SkyCoord
                processed['coordinates'] = SkyCoord(
                    ra=processed.pop('ra'),
                    dec=processed.pop('dec'),
                    unit=(u.deg, u.deg)
                )
            
            # Convert radius/size to astropy quantity if it's not already
            if 'radius' in processed and not hasattr(processed['radius'], 'unit'):
                processed['radius'] = processed.pop('radius') * u.deg
            
            # Some services use 'size' instead of 'radius'
            if 'size' in processed and not hasattr(processed['size'], 'unit'):
                processed['size'] = processed.pop('size') * u.deg
    
        # Handle object name aliases
        if query_type == 'query_object':
            # This is a common parameter name in astroquery
            target_param = 'object_name' 
            for alias in ['objectname', 'target', 'source']:
                if alias in processed:
                    processed[target_param] = processed.pop(alias)
                    break
        
        # Handle SQL query parameter mapping
        if query_type == 'query_sql' or 'sql' in query_type:
            # SDSS and some other services use 'sql_query' instead of 'sql'
            if 'sql' in processed and 'sql_query' not in processed:
                processed['sql_query'] = processed.pop('sql')
        
        return processed

    def _process_results(self, result, service_name, query_type, kwargs, auto_save):
        """Standardize query results and handle auto-saving."""
        data = None
        num_rows = 0
        save_result = None

        logger.info(f"Processing result of type: {type(result)} for service {service_name}")

        def clean_value(value):
            """Converts numpy/special types to standard python types for JSON."""
            if isinstance(value, (np.integer, np.int64)):
                return int(value)
            if isinstance(value, (np.floating, np.float32, np.float64)):
                return float(value)
            if isinstance(value, bytes):
                return value.decode('utf-8', 'ignore')
            return value
        
        def process_row(row):
            """Safely processes a dict or an astropy.table.Row into a clean dict."""
            if isinstance(row, dict):
                return {k: clean_value(v) for k, v in row.items()}
            # astropy.table.Row can be accessed by column name
            elif hasattr(row, 'colnames'):
                return {col: clean_value(row[col]) for col in row.colnames}
            # Handle other potential non-row items in a list
            return clean_value(row)

        if isinstance(result, Table):
            data = [process_row(row) for row in result]
            num_rows = len(data)

            if auto_save and num_rows > 0:
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"astroquery_{service_name}_{query_type}_{timestamp}.csv"
                full_path = self.source_dir / filename
                
                # Save to CSV
                result.write(full_path, format='csv', overwrite=True)
                
                # Register file
                description = f"Results from astroquery service '{service_name}' using '{query_type}'"
                serializable_kwargs = {k: str(v) for k, v in kwargs.items()}
                save_result = self._register_file(
                    filename=str(full_path),
                    description=description,
                    file_type='csv',
                    metadata={'service': service_name, 'query_type': query_type, 'query_params': serializable_kwargs}
                )
        
        # Handle a list of FITS images
        elif isinstance(result, list) and len(result) > 0 and hasattr(result[0], 'writeto'):
            logger.info(f"Detected a list of {len(result)} FITS-like objects.")
            num_rows = len(result)
            data = f"Returned {len(result)} FITS image(s)."

            if auto_save:
                saved_files_info = []
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                for i, hdu in enumerate(result):
                    filename = f"astroquery_{service_name}_{query_type}_{timestamp}_{i}.fits"
                    full_path = self.source_dir / filename
                    hdu.writeto(full_path, overwrite=True)
                    
                    description = f"FITS image {i+1}/{len(result)} from service '{service_name}'"
                    serializable_kwargs = {k: str(v) for k, v in kwargs.items()}
                    file_info = self._register_file(
                        filename=str(full_path),
                        description=description,
                        file_type='fits',
                        metadata={'service': service_name, 'query_type': query_type, 'query_params': serializable_kwargs}
                    )
                    saved_files_info.append(file_info)

                if saved_files_info:
                    first_file = saved_files_info[0]
                    save_result = {
                        'status': 'success',
                        'file_id': first_file['id'],
                        'filename': f"{len(saved_files_info)} files saved in {self.source_dir}",
                        'size_bytes': sum(f['size_bytes'] for f in saved_files_info),
                        'location': str(self.source_dir)
                    }

        # Handle a single FITS image
        elif hasattr(result, 'writeto'): # Heuristic for FITS HDU objects
            logger.info("Detected a single FITS-like object.")
            num_rows = 1
            data = "FITS image data" # Placeholder text
            
            if auto_save:
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"astroquery_{service_name}_{query_type}_{timestamp}.fits"
                full_path = self.source_dir / filename
                
                # Save to FITS
                result.writeto(full_path, overwrite=True)
                
                # Register file
                description = f"FITS image from astroquery service '{service_name}' using '{query_type}'"
                serializable_kwargs = {k: str(v) for k, v in kwargs.items()}
                save_result = self._register_file(
                    filename=str(full_path),
                    description=description,
                    file_type='fits',
                    metadata={'service': service_name, 'query_type': query_type, 'query_params': serializable_kwargs}
                )

        elif result is None:
            data = []
            num_rows = 0
        elif isinstance(result, list):
            data = [process_row(row) for row in result]
            num_rows = len(data)
        else:
            # For other types, just represent them as a string
            data = str(result)
            num_rows = 1 if data else 0

        # Make kwargs serializable for the response
        serializable_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (u.Quantity, SkyCoord)):
                serializable_kwargs[k] = str(v)
            else:
                serializable_kwargs[k] = v

        return {
            'status': 'success',
            'service': service_name,
            'query_type': query_type,
            'query_params': serializable_kwargs,
            'num_results': num_rows,
            'results': data,
            'save_result': save_result
        }

    def _register_file(self, filename: str, description: str, file_type: str, metadata: dict) -> dict:
        """
        Registers a new file with the central registry and updates statistics.
        This is a core piece of functionality that should be in the base class.
        """
        try:
            file_path = Path(filename)
            file_id = str(uuid.uuid4())
            file_stat = file_path.stat()
            
            record = {
                'id': file_id,
                'source': self.source_name,
                'filename': filename,
                'description': description,
                'file_type': file_type,
                'size_bytes': file_stat.st_size,
                'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                'metadata': metadata
            }
            
            # Use the shared registry
            self.registry['files'][file_id] = record
            
            # Update statistics
            self.registry['statistics']['total_files'] += 1
            self.registry['statistics']['total_size_bytes'] += file_stat.st_size
            self.registry['statistics']['by_source'][self.source_name] = self.registry['statistics']['by_source'].get(self.source_name, 0) + 1
            self.registry['statistics']['by_type'][file_type] = self.registry['statistics']['by_type'].get(file_type, 0) + 1
            
            return {'status': 'success', 'file_id': file_id, **record}
        
        except Exception as e:
            logger.error(f"Failed to register file {filename}: {e}")
            return {'status': 'error', 'error': str(e)}

    def _generate_auth_required_help(self, service_name: str, query_type: str, kwargs: dict) -> dict:
        """Generate a standardized response for services that require authentication."""
        service_class_name = self._services[service_name]['class'].__name__
        
        # Build the example script
        script_lines = [
            f"from astroquery.{service_name} import {service_class_name}",
            "import astropy.units as u",
            "from astropy.coordinates import SkyCoord",
            "",
            "# --- Step 1: Login ---",
            f"service = {service_class_name}()",
            "# Replace with your actual credentials",
            "service.login('your_username')",
            "",
            "# --- Step 2: Prepare Query Parameters ---"
        ]

        # Reconstruct kwargs for the example
        param_lines = []
        for key, value in kwargs.items():
            if key == 'auto_save': continue # Not part of the astroquery call
            if isinstance(value, str):
                param_lines.append(f"    {key}='{value}'")
            else:
                param_lines.append(f"    {key}={value}")
        
        param_str = ",\n".join(param_lines)
        
        # Use auto-detected query type if needed
        final_query_type = query_type
        if final_query_type == 'auto':
            try:
                final_query_type = self._detect_query_type(service_name, kwargs)
            except ValueError:
                final_query_type = "[could not auto-detect, please specify]"

        script_lines.append(f"# --- Step 3: Run Query ---")
        script_lines.append(f"results = service.{final_query_type}(")
        script_lines.append(param_str)
        script_lines.append(")")
        script_lines.append("")
        script_lines.append("print(results)")

        script_text = "\\n".join(script_lines)
        help_text = (
            f"AUTHENTICATION REQUIRED for service '{service_name}'.\\n\\n"
            "This service requires a login, and automatic authentication is not yet implemented in this tool.\\n"
            "To proceed, please run the following Python code in your own environment with your credentials:\\n\\n"
            "-------------------- PYTHON SCRIPT --------------------\\n"
            f"{script_text}\\n"
            "-------------------------------------------------------"
        )
        
        return {
            'status': 'auth_required',
            'help': help_text
        }

    def _generate_error_help(self, service_name: str, query_type: str, exception: Exception) -> str:
        """Generate helpful error messages."""
        try:
            service_details = self.get_service_details(service_name)
            capabilities = service_details.get('capabilities', [])
            examples = service_details.get('example_queries', [])
            
            help_text = f"The query failed for service '{service_name}' while attempting method '{query_type}'.\n"
            help_text += f"Error: {exception}\n\n"
            help_text += f"Available query methods for this service are: {', '.join(capabilities)}\n"
            
            if examples:
                help_text += "Here are some example queries for this service:\n"
                for ex in examples:
                    help_text += f"- {ex['description']}: `{ex['query']}`\n"
            
            help_text += "\nTip: Ensure your parameters match the requirements of the query method. You can specify a `query_type` directly to bypass auto-detection."
            return help_text
        except Exception as e:
            return f"An error occurred while generating help: {e}"

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