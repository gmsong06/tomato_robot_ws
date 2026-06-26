#!/usr/bin/env python3

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def main():
    parser = argparse.ArgumentParser(
        description="Move a single STS3215 motor."
    )

    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument(
        "--position",
        type=int,
        required=True,
        help="Target encoder position (0-4095)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=200,
        help="Moving speed",
    )

    args = parser.parse_args()

    motor_name = "motor"

    bus = FeetechMotorsBus(
        port=args.port,
        motors={
            motor_name: Motor(
                args.id,
                "sts3215",
                MotorNormMode.RANGE_0_100,
            )
        },
    )

    print(f"Connecting to {args.port}...")
    bus.connect(handshake=False)

    try:
        print(f"Enabling torque on motor {args.id}")
        bus.enable_torque(motor_name)

        print(f"Setting speed to {args.speed}")
        bus.write(
            "Moving_Speed",
            motor_name,
            args.speed,
        )

        print(f"Moving motor {args.id} to {args.position}")
        bus.write(
            "Goal_Position",
            motor_name,
            args.position,
        )

        while True:
            pos = bus.read(
                "Present_Position",
                motor_name,
                normalize=False,
            )

            error = abs(pos - args.position)

            print(
                f"\rPosition: {int(pos):4d}   Error: {int(error):4d}",
                end="",
                flush=True,
            )

            if error < 10:
                break

            time.sleep(0.02)

        print("\nReached target.")

    finally:
        try:
            bus.disable_torque(motor_name)
        except Exception:
            pass

        try:
            bus.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()