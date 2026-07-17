from __future__ import annotations

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
        estimated_surface_base: Point3D,
    ) -> tuple[CartesianWaypoint, ...]:
        contact_position = Point3D(
            x_m=(
                estimated_surface_base.x_m
                - self.config.contact_standoff_m
            ),
            y_m=(
                estimated_surface_base.y_m
                + self.config.contact_lateral_offset_m
            ),
            z_m=(
                estimated_surface_base.z_m
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
            position = waypoint.position_base
            ik_result = self.ik_solver.solve(
                position.x_m,
                position.y_m,
                position.z_m,
                tool_angle_from_horizontal=waypoint.tool_angle_rad,
                elbow_solution=self.config.elbow_configuration,
                target_is_tool_tip=True,
            )

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
