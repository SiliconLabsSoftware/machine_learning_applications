import argparse
import io
import os
import time

import display_serial_core


class Reader(io.IOBase):
    def __init__(self, path, timeout=2.0):
        self.timeout = timeout
        self.path = os.path.abspath(path)
        while not os.path.exists(self.path):
            print(f"Waiting for '{path}' to be created by BLE writer...")
            time.sleep(1)
        self.pos = os.path.getsize(self.path)  # skip stale data
        print(f"Opened '{path}' for reading (skipped {self.pos} bytes of stale data)")
        print("Waiting for new data from BLE device...")

    def _wait_for_data(self, n_bytes, deadline):
        """Block until at least n_bytes are available past self.pos, or deadline."""
        while True:
            try:
                file_size = os.path.getsize(self.path)
            except OSError:
                file_size = 0
            # Handle truncation (writer restarted)
            if file_size < self.pos:
                print(f"WARNING: file truncated ({self.pos} > {file_size}), resetting")
                self.pos = 0
            available = file_size - self.pos
            if available >= n_bytes:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return available > 0  # return True if we have *some* data
            time.sleep(0.01)

    def _read_from_file(self, n):
        """Open, seek, read, close. Guarantees fresh data on Windows."""
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read(n)
        if data:
            self.pos += len(data)
        return data

    def read(self, n):
        data = b""
        deadline = time.monotonic() + self.timeout
        while len(data) < n:
            needed = n - len(data)
            has_data = self._wait_for_data(1, deadline)
            if not has_data:
                return data  # timeout
            chunk = self._read_from_file(needed)
            if chunk:
                data += chunk
                deadline = time.monotonic() + self.timeout
            else:
                if time.monotonic() >= deadline:
                    return data
        return data

    def readline(self):
        line = b""
        deadline = time.monotonic() + self.timeout
        while True:
            has_data = self._wait_for_data(1, deadline)
            if not has_data:
                return line  # timeout
            ch = self._read_from_file(1)
            if ch:
                line += ch
                deadline = time.monotonic() + self.timeout
                if ch == b"\n":
                    break
            else:
                if time.monotonic() >= deadline:
                    return line
        return line

    def read_chunk(self, max_bytes=4096):
        """Read whatever is currently available, up to max_bytes."""
        deadline = time.monotonic() + self.timeout
        has_data = self._wait_for_data(1, deadline)
        if not has_data:
            return b""
        # Read whatever is available
        try:
            available = os.path.getsize(self.path) - self.pos
        except OSError:
            return b""
        to_read = min(max_bytes, max(available, 1))
        return self._read_from_file(to_read)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    display_serial_core.add_args(parser)
    args = parser.parse_args()

    vusb_path = "vusb"
    ser = Reader(vusb_path, timeout=2.0)

    try:
        display_serial_core.display_serial(ser, args)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
