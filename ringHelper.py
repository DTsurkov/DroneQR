import sys
import glob
import serial
import time


def serial_ports():
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


def get_serial():
    ports = serial_ports()
    return (list(filter(lambda ports: "Bluetooth" not in ports, ports)))[0]


if __name__ == '__main__':
    print(get_serial())
    ser = serial.Serial(get_serial(), 115200)
    ser.write(bytes("CRCR", 'utf-8'))
    time.sleep(0.5)
    ser.close()