# python scripts/jog_motor.py --id 1

import argparse
import select
import sys
import termios
import time
import tty

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


def read_key_nonblocking():
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if readable:
        return sys.stdin.read(1)
    return None


def clamp(value, low, high):
    return max(low, min(high, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--step-small", type=int, default=10)
    parser.add_argument("--step-large", type=int, default=100)
    parser.add_argument("--min-position", type=int, default=0)
    parser.add_argument("--max-position", type=int, default=4095)
    parser.add_argument("--rate", type=float, default=20.0)
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

    print(f"Connecting to motor {args.id} on {args.port}...")
    bus.connect(handshake=False)

    old_settings = termios.tcgetattr(sys.stdin)

    try:
        bus.disable_torque(motor_name)

        bus.write(
            "Operating_Mode",
            motor_name,
            OperatingMode.POSITION.value,
            normalize=False,
        )

        bus.enable_torque(motor_name)

        current_pos = int(
            bus.read(
                "Present_Position",
                motor_name,
                normalize=False,
            )
        )

        target_pos = current_pos

        print(
            f"""
Jog motor {args.id}

Controls:
  a : -{args.step_small} ticks
  d : +{args.step_small} ticks
  A : -{args.step_large} ticks
  D : +{args.step_large} ticks
  p : print current position
  q : quit

Starting position: {current_pos}
"""
        )

        tty.setcbreak(sys.stdin.fileno())

        dt = 1.0 / args.rate

        while True:
            key = read_key_nonblocking()

            if key == "a":
                target_pos -= args.step_small
            elif key == "d":
                target_pos += args.step_small
            elif key == "A":
                target_pos -= args.step_large
            elif key == "D":
                target_pos += args.step_large
            elif key == "p":
                current_pos = int(
                    bus.read(
                        "Present_Position",
                        motor_name,
                        normalize=False,
                    )
                )
                print(f"\nCurrent position: {current_pos}")
            elif key == "q":
                print("\nQuit requested")
                break

            target_pos = clamp(
                target_pos,
                args.min_position,
                args.max_position,
            )

            bus.write(
                "Goal_Position",
                motor_name,
                int(target_pos),
                normalize=False,
            )

            current_pos = int(
                bus.read(
                    "Present_Position",
                    motor_name,
                    normalize=False,
                )
            )

            print(
                f"\rTarget: {target_pos:4d} | Current: {current_pos:4d} | Error: {target_pos - current_pos:+5d}",
                end="",
                flush=True,
            )

            time.sleep(dt)

    finally:
        print("\nStopping motor.")
        try:
            bus.disable_torque(motor_name)
        except Exception:
            pass

        try:
            bus.disconnect()
        except Exception:
            pass

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()