# python ~/tomato_robot_ws/src/tomato_motor_control/scripts/change_motor_id.py \
#     --port /dev/ttyACM0 \
#     --old-id 6 \
#     --new-id 10

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def main():
    parser = argparse.ArgumentParser(
        description="Change the ID of a Feetech STS3215 servo."
    )

    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--old-id", type=int, required=True)
    parser.add_argument("--new-id", type=int, required=True)

    args = parser.parse_args()

    motor_name = "motor"

    bus = FeetechMotorsBus(
        port=args.port,
        motors={
            motor_name: Motor(
                args.old_id,
                "sts3215",
                MotorNormMode.RANGE_0_100,
            )
        },
    )

    print(f"Connecting to {args.port}...")
    bus.connect(handshake=False)

    try:
        print(f"Changing motor ID {args.old_id} -> {args.new_id}")

        bus.write(
            "ID",
            motor_name,
            args.new_id,
        )

        print("Done.")

    finally:
        try:
            bus.port_handler.closePort()
        except Exception:
            pass

    print("\nVerifying...")

    verify_bus = FeetechMotorsBus(
        port=args.port,
        motors={
            motor_name: Motor(
                args.new_id,
                "sts3215",
                MotorNormMode.RANGE_0_100,
            )
        },
    )

    verify_bus.connect(handshake=False)

    try:
        pos = verify_bus.read(
            "Present_Position",
            motor_name,
            normalize=False,
        )

        print(f"Success! Motor now responds as ID {args.new_id}")
        print(f"Current position: {int(pos)}")

    finally:
        verify_bus.disconnect()


if __name__ == "__main__":
    main()