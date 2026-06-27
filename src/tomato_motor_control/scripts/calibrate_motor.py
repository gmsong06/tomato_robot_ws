# python ~/tomato_robot_ws/src/tomato_motor_control/scripts/calibrate_motor.py   --motor joint_1:1   --motor joint_2:2   --motor joint_3:3   --motor joint_4:4 

import argparse
from pathlib import Path

import yaml
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


def parse_motor_arg(arg: str):
    if ":" not in arg:
        raise argparse.ArgumentTypeError(
            f"Invalid motor format '{arg}'. Use joint_name:id, e.g. joint_1:1"
        )

    name, motor_id = arg.split(":", 1)
    return name.strip(), int(motor_id)


def wait_enter(prompt):
    input(f"\n{prompt}\nPress ENTER when ready...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument(
        "--motor",
        action="append",
        type=parse_motor_arg,
        required=True,
        help="Example: --motor joint_1:1",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path.home()
            / "tomato_robot_ws/src/tomato_motor_control/config/motors.yaml"
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
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    config["motors"] = config.get("motors", {})

    try:
        print("\nDisabling torque on all motors...")
        bus.disable_torque()

        print("\nSetting all motors to POSITION mode...")
        for joint_name in motors.keys():
            bus.write(
                "Operating_Mode",
                joint_name,
                OperatingMode.POSITION.value,
                normalize=False,
            )

        wait_enter(
            "Move the whole arm to the MIDDLE of its safe range of motion.\n"
            "This should be a comfortable neutral pose, not near any joint limit."
        )

        print("\nSetting half-turn homings...")
        homing_offsets = bus.set_half_turn_homings()

        print("\nHoming offsets:")
        for joint_name, offset in homing_offsets.items():
            print(f"  {joint_name}: {offset}")

        print(
            "\nNow move each joint through its full SAFE range of motion.\n"
            "Move slowly. Do not force hard stops.\n"
            "Press ENTER when you have moved all joints through their ranges."
        )

        range_mins, range_maxes = bus.record_ranges_of_motion()

        print("\nMeasured ranges:")
        for joint_name in motors.keys():
            print(
                f"  {joint_name}: "
                f"min={range_mins[joint_name]}, "
                f"max={range_maxes[joint_name]}"
            )

        for joint_name, motor_id in motor_specs:
            config["motors"][joint_name] = {
                "id": motor_id,
                "model": "sts3215",
                "homing_offset": int(homing_offsets[joint_name]),
                "range_min": int(range_mins[joint_name]),
                "range_max": int(range_maxes[joint_name]),
            }

        with open(output_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        print(f"\nCalibration saved to: {output_path}")
        print(yaml.safe_dump(config, sort_keys=False))

    finally:
        try:
            bus.disable_torque()
        except Exception:
            pass

        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()