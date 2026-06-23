
import argparse
from pathlib import Path

import yaml
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def parse_motor_arg(arg: str):
    """
    Parse joint_name:id, e.g. joint_1:1
    """
    if ":" not in arg:
        raise argparse.ArgumentTypeError(
            f"Invalid motor format '{arg}'. Use joint_name:id, e.g. joint_1:1"
        )

    name, motor_id = arg.split(":", 1)
    name = name.strip()

    if not name:
        raise argparse.ArgumentTypeError("Joint name cannot be empty")

    try:
        motor_id = int(motor_id)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Motor id must be int, got '{motor_id}'")

    return name, motor_id


def read_position(bus, joint_name):
    pos = bus.read("Present_Position", joint_name, normalize=False)
    return int(pos)


def wait_enter(prompt):
    input(f"\n{prompt}\nPress ENTER when ready...")


def calibrate_joint(bus, joint_name, motor_id):
    print("\n" + "=" * 60)
    print(f"Calibrating {joint_name} | motor id {motor_id}")
    print("=" * 60)

    try:
        bus.disable_torque(joint_name)
    except Exception as e:
        print(f"Warning: could not disable torque for {joint_name}: {e}")

    print("\nTorque should be disabled.")
    print("Move the servo slowly by hand.")
    print("Do NOT force the motor into hard stops.")

    wait_enter(f"Move {joint_name} to ZERO position.")
    zero_tick = read_position(bus, joint_name)
    print(f"{joint_name} zero_tick = {zero_tick}")

    wait_enter(f"Move {joint_name} to MIN safe position.")
    min_tick = read_position(bus, joint_name)
    print(f"{joint_name} min_tick = {min_tick}")

    wait_enter(f"Move {joint_name} to MAX safe position.")
    max_tick = read_position(bus, joint_name)
    print(f"{joint_name} max_tick = {max_tick}")

    # Optional cleanup: store min/max ordered numerically
    low_tick = min(min_tick, max_tick)
    high_tick = max(min_tick, max_tick)

    print("\nResult:")
    print(f"{joint_name}:")
    print(f"  id: {motor_id}")
    print(f"  zero_tick: {zero_tick}")
    print(f"  min_tick: {low_tick}")
    print(f"  max_tick: {high_tick}")

    return {
        "id": motor_id,
        "zero_tick": zero_tick,
        "min_tick": low_tick,
        "max_tick": high_tick,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate one or more Feetech/SO-ARM servos."
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument(
        "--motor",
        action="append",
        type=parse_motor_arg,
        required=True,
        help="Motor mapping as joint_name:id. Example: --motor joint_1:1",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path.home()
            / "tomato_robot_ws/src/tomato_motor_control/config/motor_calibration.yaml"
        ),
    )

    args = parser.parse_args()

    motor_specs = args.motor

    motors = {
        joint_name: Motor(
            motor_id,
            "sts3215",
            MotorNormMode.RANGE_0_100,
        )
        for joint_name, motor_id in motor_specs
    }

    print(f"Connecting on port {args.port}...")
    print("Motors:")
    for joint_name, motor_id in motor_specs:
        print(f"  {joint_name}: id {motor_id}")

    bus = FeetechMotorsBus(
        port=args.port,
        motors=motors,
    )

    bus.connect(handshake=False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path, "r") as f:
            calibration = yaml.safe_load(f) or {}
    else:
        calibration = {}

    try:
        for joint_name, motor_id in motor_specs:
            calibration[joint_name] = calibrate_joint(
                bus=bus,
                joint_name=joint_name,
                motor_id=motor_id,
            )

            with open(output_path, "w") as f:
                yaml.safe_dump(calibration, f, sort_keys=False)

            print(f"\nSaved partial calibration to: {output_path}")

        print("\n" + "=" * 60)
        print("Calibration complete")
        print("=" * 60)
        print(yaml.safe_dump(calibration, sort_keys=False))

    finally:
        for joint_name, _ in motor_specs:
            try:
                bus.disable_torque(joint_name)
            except Exception:
                pass

        try:
            bus.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()