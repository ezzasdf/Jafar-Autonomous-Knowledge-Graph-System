"""
GeometryEngine — Parametric 3D geometry generation for Jafar.

Creates STL mesh files for mechanical parts using trimesh + numpy-stl.
Supports gear profiles, drone frames, and custom design specs.

Each generation is stored as a design experience for future reference.
"""

import json
import logging
import math
import os
import time
import uuid
from typing import Dict, List, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

GEOMETRY_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "geometry"
)


class GeometryEngine:
    """Generates 3D geometry for mechanical parts, exports to STL.

    Maintains a history of generated designs for later reference
    (experience storage).
    """

    def __init__(self, output_dir: str = GEOMETRY_OUTPUT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._experiences: List[Dict[str, Any]] = []
        logger.info("GeometryEngine initialized (output=%s)", output_dir)

    def generate_gear(self, teeth: int = 20, module: float = 1.0,
                      thickness: float = 5.0,
                      bore_diameter: float = 3.0,
                      pressure_angle: float = 20.0) -> Dict[str, Any]:
        """Generate a spur gear as STL mesh.

        Args:
            teeth: Number of gear teeth (>= 3)
            module: Gear module (mm per tooth)
            thickness: Gear face width in mm
            bore_diameter: Center bore diameter in mm
            pressure_angle: Standard pressure angle in degrees (14.5 or 20)

        Returns:
            Dict with: success, filepath, stats (pitch_diameter, etc.)
        """
        start = time.time()

        if teeth < 3:
            return {"success": False, "error": f"teeth={teeth} must be >= 3"}
        if module <= 0:
            return {"success": False, "error": f"module={module} must be > 0"}
        if thickness <= 0:
            return {"success": False, "error": f"thickness={thickness} must be > 0"}

        pitch_diameter = teeth * module
        addendum = module
        dedendum = 1.25 * module
        outer_radius = pitch_diameter / 2 + addendum
        root_radius = pitch_diameter / 2 - dedendum
        base_radius = pitch_diameter / 2 * math.cos(math.radians(pressure_angle))

        angular_pitch = 2 * math.pi / teeth
        half_tooth_angle = angular_pitch / 4

        points: List[float] = []

        for i in range(teeth):
            start_angle = i * angular_pitch
            tip_angle = start_angle + half_tooth_angle
            next_start = (i + 1) * angular_pitch - half_tooth_angle

            # Root point
            points.append((root_radius * math.cos(start_angle),
                           root_radius * math.sin(start_angle)))
            # Approach flank (involute approximation via straight line)
            point_r = root_radius
            steps = 5
            for s in range(1, steps + 1):
                frac = s / steps
                r = root_radius + (outer_radius - root_radius) * frac
                if r < base_radius:
                    r = base_radius
                angle = start_angle + half_tooth_angle * frac
                points.append((r * math.cos(angle), r * math.sin(angle)))
            # Tip arc
            tip_left = start_angle + half_tooth_angle * 0.7
            tip_right = next_start - half_tooth_angle * 0.7
            for t_frac in range(1, 4):
                a = tip_left + (tip_right - tip_left) * t_frac / 4
                points.append((outer_radius * math.cos(a),
                               outer_radius * math.sin(a)))
            # Retreat flank (mirror of approach)
            for s in range(steps, 0, -1):
                frac = s / steps
                r = root_radius + (outer_radius - root_radius) * frac
                if r < base_radius:
                    r = base_radius
                angle = next_start - half_tooth_angle * frac
                points.append((r * math.cos(angle), r * math.sin(angle)))
            # Next root
            points.append((root_radius * math.cos(next_start),
                           root_radius * math.sin(next_start)))

        try:
            import trimesh
            from shapely.geometry import Polygon
        except ImportError as e:
            return {"success": False, "error": f"Missing dependency: {e}"}

        poly = Polygon(points)
        if not poly.is_valid:
            return {"success": False, "error": "Generated gear profile is not a valid polygon"}

        mesh = trimesh.creation.extrude_polygon(poly, thickness)

        # Cut center bore
        if bore_diameter > 0:
            bore = trimesh.primitives.Cylinder(
                radius=bore_diameter / 2,
                height=thickness * 1.5,
                sections=24,
            )
            try:
                mesh = mesh.difference(bore)
            except Exception:
                # Fallback: subtract bore via trimesh boolean
                try:
                    mesh = trimesh.boolean.difference([mesh, bore])
                except Exception as e:
                    logger.warning("Bore cut failed: %s — exporting without bore", e)

        pid = uuid.uuid4().hex[:8]
        filepath = os.path.join(self.output_dir, f"gear_{teeth}t_{module}mod_{pid}.stl")

        mesh.export(filepath)
        elapsed = time.time() - start

        stats = {
            "teeth": teeth,
            "module": module,
            "thickness": thickness,
            "pitch_diameter": round(pitch_diameter, 2),
            "outer_diameter": round(outer_radius * 2, 2),
            "bore_diameter": bore_diameter,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "volume": round(mesh.volume, 2) if hasattr(mesh, 'volume') else 0,
        }

        self._experiences.append({
            "type": "gear",
            "params": {"teeth": teeth, "module": module, "thickness": thickness},
            "filepath": filepath,
            "stats": stats,
            "elapsed": round(elapsed, 3),
        })

        logger.info("Generated gear: %d teeth, module=%.1f, pitch=%.1fmm (%.2fs)",
                     teeth, module, pitch_diameter, elapsed)

        return {"success": True, "filepath": filepath, "stats": stats}

    def generate_drone_frame(self, arm_length: float = 50.0,
                             arm_width: float = 10.0,
                             thickness: float = 3.0,
                             hub_diameter: float = 20.0,
                             motor_mount_diameter: float = 5.0) -> Dict[str, Any]:
        """Generate a 4-arm X-shaped drone frame as STL mesh.

        Args:
            arm_length: Length of each arm from center to tip (mm)
            arm_width: Width of each arm cross-section (mm)
            thickness: Frame thickness (mm)
            hub_diameter: Central hub diameter (mm)
            motor_mount_diameter: Motor mount hole diameter at arm tips (mm)

        Returns:
            Dict with: success, filepath, stats
        """
        start = time.time()

        if arm_length <= 0:
            return {"success": False, "error": f"arm_length={arm_length} must be > 0"}
        if arm_width <= 0:
            return {"success": False, "error": f"arm_width={arm_width} must be > 0"}
        if thickness <= 0:
            return {"success": False, "error": f"thickness={thickness} must be > 0"}

        try:
            import trimesh
        except ImportError as e:
            return {"success": False, "error": f"Missing dependency: {e}"}

        meshes: List[Any] = []

        # Central hub
        hub = trimesh.primitives.Cylinder(
            radius=hub_diameter / 2,
            height=thickness,
            sections=24,
        )
        meshes.append(hub)

        # Four arms at 0, 90, 180, 270 degrees
        arm_radius = arm_width / 2
        arm_stub_length = arm_length - arm_radius

        for angle_deg in [0, 90, 180, 270]:
            arm = trimesh.primitives.Cylinder(
                radius=arm_radius,
                height=arm_stub_length,
                sections=16,
            )
            angle_rad = math.radians(angle_deg)
            # Build 4x4 rotation matrix around Z
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            tf = np.array([
                [cos_a, -sin_a, 0, 0],
                [sin_a,  cos_a, 0, 0],
                [0,      0,     1, 0],
                [0,      0,     0, 1],
            ])
            arm_center = np.array([
                (hub_diameter / 4 + arm_stub_length / 2) * cos_a,
                (hub_diameter / 4 + arm_stub_length / 2) * sin_a,
                0,
            ])
            arm.apply_transform(tf)
            arm.apply_translation(arm_center)
            meshes.append(arm)

        try:
            combined = trimesh.util.concatenate(meshes)
        except Exception:
            combined = meshes[0]
            for m in meshes[1:]:
                try:
                    combined = trimesh.boolean.union([combined, m])
                except Exception:
                    pass

        # Subtract motor mount holes at arm tips
        if motor_mount_diameter > 0:
            for angle_deg in [0, 90, 180, 270]:
                hole = trimesh.primitives.Cylinder(
                    radius=motor_mount_diameter / 2,
                    height=thickness * 1.5,
                    sections=12,
                )
                tip_x = arm_length * math.cos(math.radians(angle_deg))
                tip_y = arm_length * math.sin(math.radians(angle_deg))
                hole.apply_translation([tip_x, tip_y, 0])
                try:
                    combined = combined.difference(hole)
                except Exception:
                    try:
                        combined = trimesh.boolean.difference([combined, hole])
                    except Exception:
                        pass

        pid = uuid.uuid4().hex[:8]
        filepath = os.path.join(
            self.output_dir, f"drone_frame_{int(arm_length)}x{int(arm_width)}_{pid}.stl"
        )

        combined.export(filepath)
        elapsed = time.time() - start

        stats = {
            "arm_length": arm_length,
            "arm_width": arm_width,
            "thickness": thickness,
            "hub_diameter": hub_diameter,
            "vertices": len(combined.vertices),
            "faces": len(combined.faces),
            "volume": round(combined.volume, 2) if hasattr(combined, 'volume') else 0,
        }

        self._experiences.append({
            "type": "drone_frame",
            "params": {"arm_length": arm_length, "arm_width": arm_width},
            "filepath": filepath,
            "stats": stats,
            "elapsed": round(elapsed, 3),
        })

        logger.info("Generated drone frame: arms=%.1fx%.1f, hub=%.1fmm (%.2fs)",
                     arm_length, arm_width, hub_diameter, elapsed)

        return {"success": True, "filepath": filepath, "stats": stats}

    def generate_design_to_stl(self, design_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate geometry from a design specification dict.

        The spec must have a 'type' field: 'gear' or 'drone_frame'.
        Remaining fields are passed as kwargs to the specific generator.

        Returns:
            Dict from the specific generator (success, filepath, stats, or error).
        """
        design_type = design_spec.get("type", "").lower()
        params = {k: v for k, v in design_spec.items() if k != "type"}

        logger.debug("Design-to-STL: type=%s params=%s", design_type, params)

        if design_type == "gear":
            return self.generate_gear(**params)
        elif design_type == "drone_frame":
            return self.generate_drone_frame(**params)
        else:
            return {"success": False, "error": f"Unknown design type: '{design_type}'. Use 'gear' or 'drone_frame'."}

    def get_experiences(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent design experiences."""
        return self._experiences[-limit:]

    def get_experience_count(self) -> int:
        return len(self._experiences)

    def output_directory(self) -> str:
        return self.output_dir
