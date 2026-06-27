# Tomato Trajectory Package

This package implements **teach-and-repeat trajectory recording** for the tomato robot arm.

It lets you:

1. Put the robot into **recording mode**.
2. Disable motor torque.
3. Move the robot arm by hand.
4. Record the joint trajectory from `/joint_states`.
5. Save the motion to a YAML file.
6. Replay the saved motion by publishing to `/joint_target_positions`.

---

## Package Structure

```text
tomato_trajectory/
├── tomato_trajectory/
│   ├── __init__.py
│   ├── record_trajectory.py
│   └── replay_trajectory.py
├── trajectories/
├── package.xml
├── setup.py
└── README.md
```

---

## ROS Interfaces

### Subscribed Topics

```text
/joint_states
```

Used by both record and replay nodes to know the current robot joint positions.

Message type:

```text
sensor_msgs/msg/JointState
```

---

### Published Topics

```text
/joint_target_positions
```

Used by the replay node to command the robot arm.

Message type:

```text
std_msgs/msg/Float64MultiArray
```

The array should match the joint order in the trajectory file.

Example:

```text
[joint_1, joint_2, joint_3, joint_4]
```

---

### Services

```text
/set_torque
```

Used to enable or disable servo torque.

Message type:

```text
std_srvs/srv/SetBool
```

Usage:

```bash
ros2 service call /set_torque std_srvs/srv/SetBool "{data: false}"
```

```bash
ros2 service call /set_torque std_srvs/srv/SetBool "{data: true}"
```

---

## Recording a Trajectory

Start the motor node first:

```bash
ros2 launch tomato_bringup teleop.launch.py
```

Then, in a second terminal, run:

```bash
ros2 run tomato_trajectory record_trajectory --ros-args \
  -p name:=test_motion \
  -p rate_hz:=20.0
```

### What happens during recording

1. The node waits for `/joint_states`.
2. You press ENTER to start recording mode.
3. The node calls `/set_torque` with `false`.
4. The motors become free to move by hand.
5. You manually move the robot arm.
6. The node records joint positions at `rate_hz`.
7. Press ENTER to stop recording.
8. The trajectory is saved to YAML.
9. Press ENTER to re-enable torque.

---

## Replay a Trajectory

Start the motor node first:

```bash
ros2 launch tomato_bringup teleop.launch.py
```

Then run:

```bash
ros2 run tomato_trajectory replay_trajectory --ros-args \
  -p name:=test_motion \
  -p speed_scale:=1.0
```

### Replay speed

Normal speed:

```bash
-p speed_scale:=1.0
```

Twice as fast:

```bash
-p speed_scale:=2.0
```

Half speed:

```bash
-p speed_scale:=0.5
```

---

## Trajectory File Format

Recorded trajectories are saved in:

```text
tomato_trajectory/trajectories/
```

Example file:

```text
test_motion.yaml
```

Example contents:

```yaml
name: test_motion
rate_hz: 20.0
joint_names:
  - joint_1
  - joint_2
  - joint_3
  - joint_4
points:
  - t: 0.0
    positions: [0.0, 0.1, -0.2, 0.3]
  - t: 0.05
    positions: [0.01, 0.11, -0.19, 0.31]
```

Each point stores:

```text
t          timestamp in seconds
positions  joint positions in radians
```

---

## Parameters

### `record_trajectory`

| Parameter    |                                                Default | Description                            |
| ------------ | -----------------------------------------------------: | -------------------------------------- |
| `name`       |                                          `test_motion` | Name of the trajectory file            |
| `rate_hz`    |                                                 `20.0` | Recording frequency                    |
| `output_dir` | `~/tomato_robot_ws/src/tomato_trajectory/trajectories` | Directory where trajectories are saved |

Example:

```bash
ros2 run tomato_trajectory record_trajectory --ros-args \
  -p name:=pick_motion \
  -p rate_hz:=20.0
```

---

### `replay_trajectory`

| Parameter        |                                                Default | Description                                         |
| ---------------- | -----------------------------------------------------: | --------------------------------------------------- |
| `name`           |                                          `test_motion` | Name of trajectory to replay                        |
| `trajectory_dir` | `~/tomato_robot_ws/src/tomato_trajectory/trajectories` | Directory containing saved trajectories             |
| `speed_scale`    |                                                  `1.0` | Playback speed multiplier                           |
| `move_to_start`  |                                                 `true` | Move slowly to first trajectory point before replay |

Example:

```bash
ros2 run tomato_trajectory replay_trajectory --ros-args \
  -p name:=pick_motion \
  -p speed_scale:=1.0 \
  -p move_to_start:=true
```

---

## Typical Workflow

```text
1. Launch motor node
2. Record trajectory
3. Move robot by hand
4. Save trajectory
5. Replay trajectory
6. Tune speed or re-record if needed
```

Commands:

```bash
ros2 launch tomato_bringup teleop.launch.py
```

```bash
ros2 run tomato_trajectory record_trajectory --ros-args \
  -p name:=test_motion
```

```bash
ros2 run tomato_trajectory replay_trajectory --ros-args \
  -p name:=test_motion
```
