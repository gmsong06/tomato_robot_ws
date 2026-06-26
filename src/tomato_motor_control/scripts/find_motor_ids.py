# python ~/tomato_robot_ws/src/tomato_motor_control/scripts/find_motor_ids.py \
#   --port /dev/ttyACM0 \
#   --start-id 1 \
#   --end-id 20


import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=20)
    args = parser.parse_args()

    print(f"Scanning {args.port} for motor IDs {args.start_id}–{args.end_id}...")

    found = []

    for motor_id in range(args.start_id, args.end_id + 1):
        joint_name = f"motor_{motor_id}"

        motors = {
            joint_name: Motor(
                motor_id,
                "sts3215",
                MotorNormMode.RANGE_0_100,
            )
        }

        bus = FeetechMotorsBus(
            port=args.port,
            motors=motors,
        )

        try:
            bus.connect(handshake=False)

            pos = bus.read(
                "Present_Position",
                joint_name,
                normalize=False,
            )

            print(f"FOUND id={motor_id} position={int(pos)}")
            found.append(motor_id)

        except Exception:
            pass

        finally:
            try:
                bus.disconnect()
            except Exception:
                pass

    print("\nDone.")
    print(f"Found motor IDs: {found}")


if __name__ == "__main__":
    main()