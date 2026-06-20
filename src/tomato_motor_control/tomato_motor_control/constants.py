import math

TICKS_PER_REV = 4095
RAD_PER_REV = 2 * math.pi

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 1000000

def ticks_to_rad(ticks):
    return (ticks / TICKS_PER_REV) * RAD_PER_REV

def rad_to_ticks(rad):
    return int((rad / RAD_PER_REV) * TICKS_PER_REV)

def rpm_to_ticks_per_second(rpm):
    return rpm * TICKS_PER_REV / 60.0