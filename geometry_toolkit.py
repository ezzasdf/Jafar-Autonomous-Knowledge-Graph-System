"""
Jafar — Geometry Toolkit
Unified dispatcher that routes each geometry operation to the best available backend.
Uses operation-to-backend mapping (not flat priority), schema-based matching,
param normalization, and a feedback loop for learned routing.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Operation-to-backend routing map ─────────────────────────────────
# Each operation is routed to its canonical backend first.
# Wildcard "*" entries serve as category-level defaults.
# Fallback order is defined by BACKEND_FALLBACK_ORDER.

OPERATION_BACKEND_MAP: Dict[str, str] = {
    # ── CadQuery — parametric B-Rep solid modeling ──
    "parametric_box": "cadquery",
    "parametric_cylinder": "cadquery",
    "parametric_gear": "cadquery",
    "parametric_assembly": "cadquery",
    "parametric_drone": "cadquery",
    "extrude": "cadquery",
    "revolve": "cadquery",
    "cq_boolean": "cadquery",
    "export_step": "cadquery",
    # CadQuery wildcard fallback
    "parametric_*": "cadquery",
    "cq_*": "cadquery",

    # ── MeshLib — mesh repair, processing, booleans ──
    "mesh_heal": "meshlib",
    "mesh_boolean": "meshlib",
    "mesh_offset": "meshlib",
    "mesh_simplify": "meshlib",
    "mesh_smooth": "meshlib",
    "mesh_fill_holes": "meshlib",
    "mesh_inspect": "meshlib",
    "voxel_to_mesh": "meshlib",
    # MeshLib wildcard fallback
    "mesh_*": "meshlib",

    # ── OpenVDB — sparse voxel / level-set operations ──
    "mesh_to_level_set": "pyopenvdb",
    "level_set_to_mesh": "pyopenvdb",
    "vdb_boolean": "pyopenvdb",
    "vdb_morph": "pyopenvdb",
    "vdb_offset": "pyopenvdb",
    "vdb_grid_info": "pyopenvdb",
    # OpenVDB wildcard fallback
    "vdb_*": "pyopenvdb",
    "level_set_*": "pyopenvdb",

    # ── PicoGK (trimesh) — voxel / lattice / fallback ──
    "mesh_voxelize": "picogk",
    "voxel_boolean": "picogk",
    "lattice": "picogk",
    "offset": "picogk",
    "smooth": "picogk",
    "export_stl": "picogk",
    "export_obj": "picogk",
}

BACKEND_FALLBACK_ORDER: List[str] = [
    "cadquery", "meshlib", "pyopenvdb", "picogk",
]


# ── Param name translation ───────────────────────────────────────────
# canonical_name -> { backend_alias: backend_param_name }
# When normalize_params() is called, any canonical param key is mapped
# to the backend-specific key before dispatch.

PARAM_TRANSLATION: Dict[str, Dict[str, str]] = {
    "bore_diameter": {
        "cadquery": "bore_diameter",
        "meshlib": "bore",
        "picogk": "bore",
    },
    "thickness": {
        "cadquery": "thickness",
        "meshlib": "shell_thickness",
    },
}


class GeometryToolkit:
    """Unified router across all geometry backends.

    Routing strategy:
        1. Look up the operation in OPERATION_BACKEND_MAP (exact match first)
        2. Fall back to wildcard pattern (e.g. ``parametric_*``)
        3. If the primary backend is missing or fails, walk BACKEND_FALLBACK_ORDER
        4. Accept an optional ``preferred_backend`` override (e.g. from KG feedback)

    Each backend is lazy-loaded on first use.
    """

    def __init__(self):
        self._backends: Dict[str, Any] = {}
        self._routing_cache: Dict[str, str] = {}

    # ── Public API ────────────────────────────────────────────────────

    def execute(self, operation: str, params: Optional[Dict[str, Any]] = None,
                preferred_backend: str = "") -> Dict[str, Any]:
        """Execute a geometry operation, routing to the best available backend.

        Args:
            operation: The geometry operation name.
            params: Operation parameters.
            preferred_backend: Override — try this backend first.

        Returns:
            Result dict with 'success', 'filepath', 'stats', 'backend'.
        """
        params = dict(params or {})

        # Build the ordered list of backends to try
        backends_to_try = self._resolve_backend_order(operation, preferred_backend)

        for backend_name in backends_to_try:
            backend = self.get_backend(backend_name)
            if backend is None:
                continue

            if not self._backend_matches(backend, operation):
                continue

            # Normalize params for this backend
            normalized = self._normalize_params(params, backend_name)

            try:
                result = backend.run(operation=operation, params=normalized)
                if result.get("success"):
                    result["backend"] = backend_name
                    return result
            except Exception as e:
                logger.debug("Backend %s failed for %s: %s",
                             backend_name, operation, e)
                continue

        return {
            "success": False,
            "error": f"No backend could handle operation '{operation}'",
            "operation": operation,
        }

    def run_tool(self, tool_name: str, operation: str,
                 params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a specific tool by name with the given operation."""
        backend = self.get_backend(tool_name)
        if backend is None:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not available",
            }
        try:
            normalized = self._normalize_params(params or {}, tool_name)
            result = backend.run(operation=operation, params=normalized)
            if result.get("success"):
                result["backend"] = tool_name
            return result
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return {"success": False, "error": str(e)}

    def backend_for_operation(self, operation: str) -> Optional[str]:
        """Return the canonical backend name for an operation, or None."""
        if operation in self._routing_cache:
            return self._routing_cache[operation]

        # Exact match first
        if operation in OPERATION_BACKEND_MAP:
            backend = OPERATION_BACKEND_MAP[operation]
            self._routing_cache[operation] = backend
            return backend

        # Wildcard suffix match
        for pattern, backend in OPERATION_BACKEND_MAP.items():
            if pattern.endswith("*") and operation.startswith(pattern[:-1]):
                self._routing_cache[operation] = backend
                return backend

        return None

    def list_backends(self) -> Dict[str, str]:
        """Report which backends are available."""
        status = {}
        for name in BACKEND_FALLBACK_ORDER:
            backend = self.get_backend(name)
            status[name] = "available" if backend is not None else "unavailable"
        return status

    def get_backend(self, name: str) -> Any:
        """Lazy-load and return a backend by name."""
        if name in self._backends:
            return self._backends[name]

        loaders = {
            "cadquery": self._load_cadquery,
            "meshlib": self._load_meshlib,
            "pyopenvdb": self._load_openvdb,
            "picogk": self._load_picogk,
        }
        loader = loaders.get(name)
        backend = loader() if loader else None
        self._backends[name] = backend
        return backend

    # ── Routing internals ─────────────────────────────────────────────

    def _resolve_backend_order(self, operation: str,
                                preferred_backend: str = "") -> List[str]:
        """Produce the ordered list of backends to try for *operation*."""
        primary = self.backend_for_operation(operation)

        # Start with the KG-influenced preferred backend, the canonical primary,
        # or an empty list.
        if preferred_backend and preferred_backend in BACKEND_FALLBACK_ORDER:
            order = [preferred_backend]
        elif primary and primary in BACKEND_FALLBACK_ORDER:
            order = [primary]
        else:
            order = []

        # Append remaining backends in fallback order, deduplicated
        seen = set(order)
        for name in BACKEND_FALLBACK_ORDER:
            if name not in seen:
                order.append(name)
                seen.add(name)

        return order

    # ── Param normalization ──────────────────────────────────────────

    @staticmethod
    def _normalize_params(params: Dict[str, Any],
                          backend_name: str) -> Dict[str, Any]:
        """Translate canonical param names to backend-specific names."""
        if not params:
            return params

        result = dict(params)

        for canonical, mapping in PARAM_TRANSLATION.items():
            if canonical in result and backend_name in mapping:
                backend_key = mapping[backend_name]
                if backend_key != canonical:
                    result[backend_key] = result.pop(canonical)

        return result

    # ── Backend matching (schema-based) ───────────────────────────────

    @staticmethod
    def _backend_matches(backend, operation: str) -> bool:
        """Check if a backend's ``input_schema`` declares this operation.

        Uses the backend's ``input_schema.properties.operation.enum`` list
        (the authoritative set of supported operations).
        Falls back to *True* if the schema is missing or unenumerable.
        """
        schema = getattr(backend, 'input_schema', None)
        if schema is None:
            return True
        props = schema.get("properties", {})
        op_schema = props.get("operation", {})
        valid_ops = op_schema.get("enum", [])
        if not valid_ops:
            return True
        return operation in valid_ops

    # ── Lazy loaders ─────────────────────────────────────────────────

    @staticmethod
    def _load_cadquery():
        try:
            from tools.cadquery_tool import CadQueryTool
            tool = CadQueryTool()
            logger.info("GeometryToolkit: CadQuery backend loaded")
            return tool
        except Exception as e:
            logger.debug("CadQuery backend unavailable: %s", e)
            return None

    @staticmethod
    def _load_meshlib():
        try:
            from tools.meshlib_tool import MeshLibTool
            tool = MeshLibTool()
            logger.info("GeometryToolkit: MeshLib backend loaded")
            return tool
        except Exception as e:
            logger.debug("MeshLib backend unavailable: %s", e)
            return None

    @staticmethod
    def _load_openvdb():
        try:
            from tools.pyopenvdb_tool import PyOpenVDBTool
            tool = PyOpenVDBTool()
            logger.info("GeometryToolkit: PyOpenVDB backend loaded")
            return tool
        except Exception as e:
            logger.debug("PyOpenVDB backend unavailable: %s", e)
            return None

    @staticmethod
    def _load_picogk():
        try:
            from tools.picogk import PicoGKTool
            tool = PicoGKTool()
            logger.info("GeometryToolkit: PicoGKTool backend loaded")
            return tool
        except Exception as e:
            logger.debug("PicoGKTool backend unavailable: %s", e)
            return None
