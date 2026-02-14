"""
Module Manager - Allows enabling/disabling plugin features dynamically.
Supports adding/removing modules at runtime for extensibility.
"""
import logging
from typing import Dict, Any, Optional
logger = logging.getLogger("ModuleManager")

class ModuleInfo:
    def __init__(self, name: str, display_name: str, description: str,
                 instance: Any, enabled: bool = True):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.instance = instance
        self.enabled = enabled

class ModuleManager:
    """
    Central module registry. Modules can be registered, enabled, disabled,
    and queried at runtime. This enables the modular architecture requirement.
    """
    def __init__(self, config: dict):
        self._modules: Dict[str, ModuleInfo] = {}
        self._enabled_modules = set(config.get('enabled', []))
        logger.info("ModuleManager initialized")
    def register_module(self, name: str, display_name: str,
                        description: str, instance: Any):
        """Register a new module with the manager."""
        enabled = name in self._enabled_modules
        self._modules[name] = ModuleInfo(
            name=name,
            display_name=display_name,
            description=description,
            instance=instance,
            enabled=enabled
        )
        logger.info(f"Module registered: {name} (enabled={enabled})")
    def unregister_module(self, name: str):
        """Remove a module from the registry."""
        if name in self._modules:
            del self._modules[name]
            self._enabled_modules.discard(name)
            logger.info(f"Module unregistered: {name}")
    def toggle_module(self, name: str, enabled: bool) -> dict:
        """Enable or disable a module."""
        if name not in self._modules:
            return {"status": "error", "message": f"Module '{name}' not found"}
        self._modules[name].enabled = enabled
        if enabled:
            self._enabled_modules.add(name)
        else:
            self._enabled_modules.discard(name)
        logger.info(f"Module {name} {'enabled' if enabled else 'disabled'}")
        return {"status": "ok", "module": name, "enabled": enabled}
    def is_enabled(self, name: str) -> bool:
        """Check if a module is enabled."""
        mod = self._modules.get(name)
        return mod.enabled if mod else False
    def get_module(self, name: str) -> Optional[Any]:
        """Get the instance of a module."""
        mod = self._modules.get(name)
        return mod.instance if mod and mod.enabled else None
    def list_modules(self) -> list:
        """List all registered modules with their status."""
        return [
            {
                "name": m.name,
                "display_name": m.display_name,
                "description": m.description,
                "enabled": m.enabled
            }
            for m in self._modules.values()
        ]
    def get_enabled_modules(self) -> list:
        """Get all enabled module instances."""
        return [m.instance for m in self._modules.values() if m.enabled]
