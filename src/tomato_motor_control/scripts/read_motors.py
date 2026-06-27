# python ~/tomato_robot_ws/src/tomato_motor_control/scripts/read_motors.py --port /dev/ttyACM0 --ids 1 2 3 4 5 6

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--ids", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--rate", type=float, default=5.0)
    args = parser.parse_args()

    motors = {
        f"motor_{motor_id}": Motor(
            motor_id,
            "sts3215",
            MotorNormMode.RANGE_0_100,
        )
        for motor_id in args.ids
    }

    bus = FeetechMotorsBus(
        port=args.port,
        motors=motors,
    )

    print(f"Connecting to {args.port}...")
    bus.connect(handshake=False)

    try:
        for name in motors.keys():
            try:
                bus.disable_torque(name)
            except Exception:
                pass

        print("Reading motor positions.")
        print("Move one joint by hand and watch which ID changes.")
        print("Press Ctrl+C to stop.\n")

        dt = 1.0 / args.rate

        while True:
            parts = []

            for motor_id in args.ids:
                name = f"motor_{motor_id}"

                try:
                    pos = bus.read(
                        "Present_Position",
                        name,
                        normalize=False,
                    )
                    parts.append(f"id {motor_id}: {int(pos):4d}")
                except Exception as e:
                    parts.append(f"id {motor_id}: ERR")

            print(" | ".join(parts))
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        for name in motors.keys():
            try:
                bus.disable_torque(name)
            except Exception:
                pass

        try:
            bus.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()