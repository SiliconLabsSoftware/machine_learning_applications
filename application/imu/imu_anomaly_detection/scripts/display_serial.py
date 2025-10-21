import argparse
import serial
import serial.tools.list_ports
import matplotlib.pyplot as plt
import os


def display(ser):
    buffer_len = 50
    buffer = [0] * buffer_len
    fig, axs = plt.subplots(1, 1)
    while True:
        raw_line = ser.readline()
        line = raw_line.decode("utf-8").strip()
        print(line)
        if line.startswith("anom_score"):
            line_info = line.split(":")
            anomaly_score = float(line_info[1])
            # Add anomaly score to buffer
            buffer.append(anomaly_score)
            buffer = buffer[1:]
            # Plot buffer
            # Set axis limits
            axs.set_ylim(0, 1)
            axs.set_xlim(0, buffer_len)
            # Set title
            axs.set_title("Anomaly Score")
            axs.plot(buffer, color="red", linewidth=1)
            # Draw plot to screen
            fig.tight_layout()
            fig.canvas.draw()
            plt.pause(0.001)
            axs.clear()


if __name__ == "__main__":
    ports = list(serial.tools.list_ports.comports())
    for i, p in enumerate(ports):
        print(f"{i}: {p}")
        print(p.name)
    port_id = input("Select port to connect to: ")
    port_id = int(port_id)

    ser = serial.Serial(port=ports[port_id].device, timeout=0.1)

    display(ser)
