# Tomato Motor Control Scripts

This directory contains utility scripts for configuring, calibrating, testing, and teleoperating Feetech STS3215 servos used by the tomato harvesting robot.

---

# Setup

Activate the virtual environment:

```bash
tomato
```

All commands below assume the USB adapter is connected at:

```text
/dev/ttyACM0
```

---

# 1. Find Motor IDs

Scan the servo bus for connected motors.

```bash
python scripts/find_motor_ids.py \
    --port /dev/ttyACM0 \
    --start-id 1 \
    --end-id 20
```

Example output:

```text
FOUND id=1
FOUND id=2
FOUND id=3
FOUND id=4
```

---

# 2. Change a Motor ID

Assign a new ID to a motor.

```bash
python scripts/change_motor_id.py \
    --port /dev/ttyACM0 \
    --old-id 6 \
    --new-id 4
```

Verify afterwards with `find_motor_ids.py`.

---

# 3. Read Motor Positions

Continuously read encoder values.

```bash
python scripts/read_motors.py \
    --port /dev/ttyACM0
```

---


# 4 Set Operating Mode

Configure a motor's operating mode.

Examples:

Position mode

```bash
python scripts/set_operating_mode.py \
    --id 1 \
    --mode position
```

Velocity mode

```bash
python scripts/set_operating_mode.py \
    --id 1 \
    --mode velocity
```

---

# 5. Jog Motor

Interactively move one motor.

```bash
python scripts/jog_motor.py \
    --id 1
```

Controls:

```text
+   increase position
-   decrease position
q   quit
```

Useful before running calibration.

---

# 6. Calibrate Motors

Calibrate one or more joints.

Example:

```bash
python scripts/calibrate_motor.py \
    --motor joint_1:1 \
    --motor joint_2:2 \
    --motor joint_3:3 \
    --motor joint_4:4
```

Calibration procedure:

1. Torque is disabled on all motors.
2. Move the requested joint to its center position.
3. Press ENTER.
4. Move the joint through its complete range of motion.
5. Press ENTER again.
6. Calibration values are saved automatically.

Output:

```text
config/motors.yaml
```

---

# 7. Build Workspace

```bash
cbs
```

or

```bash
colcon build --symlink-install
```

Then source:

```bash
source install/setup.bash
```

---

# 8. Launch Motor Node

Start the ROS motor driver.

```bash
ros2 launch tomato_bringup teleop.launch.py
```

---

# 9. Keyboard Teleoperation

Open a second terminal.

```bash
tomato
```

Run:

```bash
ros2 run tomato_teleop keyboard_teleop_node
```

Controls:

```text
q / a   Joint 1
w / s   Joint 2
e / d   Joint 3
r / f   Joint 4

Space   Republish current target
x       Quit
```

---

# Files

```text
scripts/
├── calibrate_motor.py
├── change_motor_id.py
├── find_motor_ids.py
├── jog_motor.py
├── read_motors.py
└── set_operating_mode.py

config/
└── motors.yaml
```
