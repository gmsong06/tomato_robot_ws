# python scripts/set_operating_mode.py --id 1 --mode velocity
# python scripts/set_operating_mode.py --id 1 --mode position

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


MODES = {
    "position": OperatingMode.POSITION.value,
    "velocity": OperatingMode.VELOCITY.value,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--mode", choices=MODES.keys(), required=True)
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

    try:
        bus.disable_torque(motor_name)

        mode_value = MODES[args.mode]

        print(f"Setting motor {args.id} to {args.mode} mode ({mode_value})")

        bus.write(
            "Operating_Mode",
            motor_name,
            mode_value,
            normalize=False,
        )

        mode = bus.read(
            "Operating_Mode",
            motor_name,
            normalize=False,
        )

        print(f"Read back operating mode: {mode}")

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