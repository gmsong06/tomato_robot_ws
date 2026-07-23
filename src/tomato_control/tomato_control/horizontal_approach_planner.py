from __future__ import annotations

import math

from tomato_control.controller_config import ControllerConfig
from tomato_control.controller_models import (
    CartesianWaypoint,
    Point3D,
    WaypointCommand,
)


class HorizontalApproachPlanner:
    """Create and solve the tomato-relative pregrasp/contact/retreat path."""

    def __init__(self, config: ControllerConfig, ik_solver, logger):
        self.config = config
        self.ik_solver = ik_solver
        self.logger = logger

    def create_waypoints(
        self,
        estimated_surface_origin: Point3D,
    ) -> tuple[CartesianWaypoint, ...]:
        contact_position = Point3D(
            x_m=(
                estimated_surface_origin.x_m
                - self.config.contact_standoff_m
            ),
            y_m=(
                estimated_surface_origin.y_m
                + self.config.contact_lateral_offset_m
            ),
            z_m=(
                estimated_surface_origin.z_m
                + self.config.contact_vertical_offset_m
            ),
        )

        pregrasp_position = Point3D(
            x_m=(
                contact_position.x_m
                - self.config.pregrasp_distance_m
            ),
            y_m=contact_position.y_m,
            z_m=contact_position.z_m,
        )

        retreat_position = Point3D(
            x_m=(
                contact_position.x_m
                - self.config.retreat_distance_m
            ),
            y_m=contact_position.y_m,
            z_m=contact_position.z_m,
        )

        tool_angle = self.config.tool_angle_from_horizontal_rad
        return (
            CartesianWaypoint("pregrasp", pregrasp_position, tool_angle),
            CartesianWaypoint("contact", contact_position, tool_angle),
            CartesianWaypoint("retreat", retreat_position, tool_angle),
        )

    def solve_waypoints(
        self,
        waypoints: tuple[CartesianWaypoint, ...],
        detection_id: int,
    ) -> tuple[WaypointCommand, ...] | None:
        commands: list[WaypointCommand] = []

        for waypoint in waypoints:
            position = waypoint.position_origin
            ik_result = self._solve_waypoint(waypoint)

            if not ik_result.success:
                self.logger.warn(
                    f"id={detection_id}: IK failed for "
                    f"{waypoint.name}: {ik_result.reason}"
                )
                return None

            commands.append(
                WaypointCommand(
                    name=waypoint.name,
                    waypoint=waypoint,
                    joint_angles={
                        name: float(angle)
                        for name, angle in ik_result.joint_angles.items()
                    },
                    ik_result=ik_result,
                )
            )

        return tuple(commands)

    def _solve_waypoint(self, waypoint: CartesianWaypoint):
        """Use the exact IK call used by the normal acceptance path."""

        position = waypoint.position_origin
        return self.ik_solver.solve(
            position.x_m,
            position.y_m,
            position.z_m,
            tool_angle_from_horizontal=waypoint.tool_angle_rad,
            elbow_solution=self.config.elbow_configuration,
            target_is_tool_tip=True,
        )

    def build_ik_diagnostic_report(
        self,
        waypoints: tuple[CartesianWaypoint, ...],
    ) -> tuple[bool, str]:
        """Explain the exact waypoint/IK decisions used for reachability."""

        lines: list[str] = []
        all_passed = True
        first_failure: str | None = None
        configured_branch = self.config.elbow_configuration
        alternate_branch = "down" if configured_branch == "up" else "up"

        lines.extend(
            [
                "IK CONFIGURATION",
                f"  configured elbow branch: {configured_branch}",
                "  target interpreted as: tool tip",
                f"  upper arm length: {self.ik_solver.L1:.6f} m",
                f"  forearm length: {self.ik_solver.L2:.6f} m",
                f"  tool length: {self.ik_solver.L_tool:.6f} m",
                "  wrist-to-tool-tip offset: "
                f"({self.ik_solver.tool_offset_x:.6f}, "
                f"{self.ik_solver.tool_offset_y:.6f}, "
                f"{self.ik_solver.tool_offset_z:.6f}) m",
                "  IK target frame: fixed axes at robot origin",
                "  robot origin in URDF base_link: "
                f"({self.ik_solver.robot_origin_in_urdf_base[0]:.6f}, "
                f"{self.ik_solver.robot_origin_in_urdf_base[1]:.6f}, "
                f"{self.ik_solver.robot_origin_in_urdf_base[2]:.6f}) m",
                "  shoulder offset: "
                f"({self.ik_solver.shoulder_offset[0]:.6f}, "
                f"{self.ik_solver.shoulder_offset[1]:.6f}, "
                f"{self.ik_solver.shoulder_offset[2]:.6f}) m",
                f"  planar minimum reach: {abs(self.ik_solver.L1 - self.ik_solver.L2):.6f} m",
                f"  planar maximum reach: {self.ik_solver.L1 + self.ik_solver.L2:.6f} m",
                "",
            ]
        )

        for waypoint in waypoints:
            position = waypoint.position_origin
            authoritative = self._solve_waypoint(waypoint)
            alternate = self.ik_solver.solve(
                position.x_m,
                position.y_m,
                position.z_m,
                tool_angle_from_horizontal=waypoint.tool_angle_rad,
                elbow_solution=alternate_branch,
                target_is_tool_tip=True,
            )

            lines.extend(
                [
                    "=" * 72,
                    f"WAYPOINT: {waypoint.name.upper()}",
                    "=" * 72,
                    "Desired tool-tip target relative to robot origin:",
                    f"  x: {position.x_m:.6f} m",
                    f"  y: {position.y_m:.6f} m",
                    f"  z: {position.z_m:.6f} m",
                    "  tool angle from horizontal: "
                    f"{waypoint.tool_angle_rad:.6f} rad "
                    f"({math.degrees(waypoint.tool_angle_rad):.2f} deg)",
                    "",
                    f"Authoritative branch ({configured_branch}):",
                ]
            )
            lines.extend(self._format_ik_result(authoritative))

            lines.extend(
                [
                    "",
                    f"Alternate branch ({alternate_branch}, informational only):",
                ]
            )
            lines.extend(self._format_ik_result(alternate))
            lines.append("")

            if not authoritative.success:
                all_passed = False
                if first_failure is None:
                    first_failure = (
                        f"{waypoint.name}: {authoritative.reason}"
                    )

        lines.extend(
            [
                "=" * 72,
                "FINAL IK VERDICT",
                "=" * 72,
                f"  accepted by configured planner: {'YES' if all_passed else 'NO'}",
            ]
        )
        if first_failure is not None:
            lines.append(f"  first rejection: {first_failure}")
        else:
            lines.append("  all pregrasp, contact, and retreat waypoints passed")

        return all_passed, "\n".join(lines)

    def _format_ik_result(self, result) -> list[str]:
        lines = [
            f"  success: {'YES' if result.success else 'NO'}",
            f"  reason: {result.reason or '(none)'}",
        ]

        if result.wrist_target is not None:
            wrist_r, wrist_z = result.wrist_target
            lines.extend(
                [
                    f"  wrist target radial: {wrist_r:.6f} m",
                    f"  wrist target vertical: {wrist_z:.6f} m",
                    f"  shoulder-to-wrist distance: {math.hypot(wrist_r, wrist_z):.6f} m",
                ]
            )

        metadata = result.metadata or {}
        for key in (
            "horizontal_radius",
            "radial_to_tip",
            "tool_lateral",
            "target_bearing",
            "arm_plane_bearing",
            "yaw_zero_offset",
            "r",
            "z_rel",
            "h",
            "shoulder_geom",
            "forearm_geom",
            "tool_angle_from_horizontal",
        ):
            if key in metadata:
                lines.append(f"  {key}: {float(metadata[key]):.6f}")

        if result.joint_angles:
            lines.append("  joint solution and URDF limits:")
            for joint_name in (
                self.ik_solver.joint_1_name,
                self.ik_solver.joint_2_name,
                self.ik_solver.joint_3_name,
                self.ik_solver.joint_4_name,
            ):
                if joint_name not in result.joint_angles:
                    continue
                angle = float(result.joint_angles[joint_name])
                lower, upper = self.ik_solver.joint_limits.get(
                    joint_name,
                    (None, None),
                )
                lower_text = "-inf" if lower is None else f"{lower:.6f}"
                upper_text = "+inf" if upper is None else f"{upper:.6f}"
                within_limits = (
                    (lower is None or angle >= lower)
                    and (upper is None or angle <= upper)
                )
                lines.append(
                    f"    {joint_name}: {angle:.6f} rad "
                    f"({math.degrees(angle):.2f} deg), "
                    f"limits=[{lower_text}, {upper_text}], "
                    f"{'PASS' if within_limits else 'FAIL'}"
                )
        else:
            lines.append("  joint solution: unavailable")

        return lines
