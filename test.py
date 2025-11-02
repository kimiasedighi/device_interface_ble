import asyncio
import csv
import datetime as dt
from bleak import BleakScanner, BleakClient

# ---------------- BLE UUIDs (match firmware) ----------------
DEVICE_NAME = "Bracelet-EMG"
SVC_UUID = "12345678-1234-1234-1234-1234567890ab"
TX_UUID  = "12345678-1234-1234-1234-1234567890ac"   # Notify
RX_UUID  = "12345678-1234-1234-1234-1234567890ad"   # Write

# ---------------- CSV Logger ----------------
class CSVLogger:
    def __init__(self):
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file = open(f"emg_raw_{ts}.csv", "w", newline="", buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp_ms", "hex_data"])

    def log(self, data: bytes):
        t = int(dt.datetime.now().timestamp() * 1000)
        self.writer.writerow([t, data.hex(" ")])

    def close(self):
        try:
            self.file.flush()
        finally:
            self.file.close()

# ---------------- Main BLE logic ----------------
async def main():
    print("[INFO] Scanning for Bracelet-EMG...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=15.0)
    if not device:
        print("[ERROR] Bracelet-EMG not found! Make sure it's advertising.")
        return

    print(f"[INFO] Found {device.address}, connecting...")
    logger = CSVLogger()

    def on_notify(_, data: bytearray):
        print(f"[BLE] {len(data)} bytes: {data.hex(' ')}")
        logger.log(data)

    async with BleakClient(device) as client:
        print("[INFO] Connected!")

        # Send start acquisition command (0x0063)
        start_cmd = bytes([0x63, 0x00])  # little-endian format
        await client.write_gatt_char(RX_UUID, start_cmd, response=True)
        print("[INFO] Sent start command 0x0063")

        # Start receiving notifications
        await client.start_notify(TX_UUID, on_notify)
        print("[INFO] Notifications started. Press Ctrl+C to stop.\n")

        try:
            while True:
                await asyncio.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping stream...")
            # Send stop acquisition command (0x0062)
            stop_cmd = bytes([0x62, 0x00])
            try:
                await client.write_gatt_char(RX_UUID, stop_cmd, response=True)
                print("[INFO] Sent stop command 0x0062")
            except Exception as e:
                print(f"[WARN] Could not send stop command: {e}")
        finally:
            await client.stop_notify(TX_UUID)
            logger.close()
            print("[INFO] Disconnected & saved data to CSV.")

# ---------------- Run ----------------
if __name__ == "__main__":
    asyncio.run(main())
