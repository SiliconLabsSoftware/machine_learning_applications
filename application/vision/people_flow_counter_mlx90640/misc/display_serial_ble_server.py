import asyncio
import io
import logging
import os

from bleak import BleakClient, BleakScanner

CHAR_UUID = "16480002-0525-4ad5-b4fb-6dd83f49546b"

# ----------------- LOGGING SETUP -----------------
logging.basicConfig(
    level=logging.INFO,  # change to INFO in production
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ble-client")


# ----------------- FILE WRITER -----------------
class Writer(io.IOBase):
    def __init__(self, path):
        logger.debug(f"Initializing Writer with path={path}")
        self.fp = open(path, "w+b")
        self.pointer = 0

    def write(self, data):
        try:
            logger.debug(f"Writing {len(data)} bytes at offset {self.pointer}")
            self.fp.seek(self.pointer)
            self.fp.write(data)
            self.fp.flush()
            os.fsync(self.fp.fileno())
            self.pointer += len(data)
        except Exception:
            logger.exception("Error while writing data")

    def close(self):
        logger.debug("Closing Writer file")
        try:
            self.fp.close()
        except Exception:
            logger.exception("Error closing file")


# ----------------- NOTIFICATION HANDLER -----------------
def create_handler():
    def handle_data(sender, data: bytearray):
        logger.debug(f"Notification received from {sender}: {len(data)} bytes")
        try:
            tty_file.write(data)
        except Exception:
            logger.exception("Failed to handle incoming data")

    return handle_data


# ----------------- CLIENT -----------------
async def run_client(device):
    disconnected = asyncio.Event()

    def on_disconnect(_client):
        logger.warning("Disconnected from device.")
        disconnected.set()

    logger.info(f"Connecting to {device.name} ({device.address}) ...")

    async with BleakClient(device, disconnected_callback=on_disconnect) as client:
        logger.info("Connected.")

        handler = create_handler()

        try:
            logger.debug(f"Starting notifications for UUID={CHAR_UUID}")
            await client.start_notify(CHAR_UUID, handler)
            logger.info("Notifications started.")
        except Exception:
            logger.exception("Failed to start notifications")
            return

        try:
            await disconnected.wait()
        finally:
            logger.debug("Stopping notifications...")
            try:
                await client.stop_notify(CHAR_UUID)
                logger.info("Notifications stopped.")
            except Exception:
                logger.exception("stop_notify failed")


# ----------------- DEVICE SELECTION -----------------
async def select_device():
    logger.info("Scanning for BLE devices...")
    devices_found = await BleakScanner.discover(timeout=2)

    devices = []
    for d in devices_found:
        # logger.debug(f"Discovered device: name={d.name}, address={d.address}")
        if d.name is None:
            continue
        print(f"{len(devices)}: {d.name} - {d.address}")
        devices.append(d)

    if not devices:
        logger.warning("No devices found.")
        return None

    try:
        port_id = int(input("Select BLE device to connect to: "))
        logger.info(f"User selected device index {port_id}")
        return devices[port_id]
    except Exception:
        logger.exception("Invalid device selection")
        return None


# ----------------- MAIN LOOP -----------------
async def main():
    logger.info("Application started")

    device = await select_device()
    if device is None:
        logger.error("No valid device selected. Exiting.")
        return

    while True:
        try:
            logger.info("Starting client session")
            await run_client(device)
        except asyncio.CancelledError:
            logger.info("Application cancelled")
            raise
        except Exception:
            logger.exception("Connection error")

        logger.info("Retrying in 1 second...")
        await asyncio.sleep(1)


# ----------------- ENTRY POINT -----------------
if __name__ == "__main__":
    logger.debug("Program entry point")

    tty_file = Writer("vusb")

    try:
        asyncio.run(main())
    finally:
        logger.debug("Shutting down application")
        tty_file.close()
