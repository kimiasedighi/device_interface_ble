import asyncio
import argparse
import csv
import datetime as dt
from typing import List, Tuple, Optional
from bleak import BleakClient, BleakScanner

# Must match firmware
SVC_UUID = "12345678-1234-1234-1234-1234567890ab"
TX_UUID  = "12345678-1234-1234-1234-1234567890ac"   # notify
RX_UUID  = "12345678-1234-1234-1234-1234567890ad"   # write
DEVICE_NAME_DEFAULT = "Bracelet-EMG"

MAG24 = b"\x00\x62\x0f"   # 24-bit header: magic(3) + ts(3, BE)
MAG16 = b"\x62\x0f"       # 16-bit header: magic(2) + ts(2, BE)

# -------- Config decoding (must mirror your firmware's updateConfigurations) --------
def decode_config(cfg: int) -> Tuple[int, int, int, bool]:
    """
    Return (nch, bytes_per_sample, header_len, res_is_24bit).
    Uses your bit mapping:
      bits 12..11 = {00:2, 01:4, 10:8, 11:16}
      bit 7 = resolution {0:16-bit, 1:24-bit}
    """
    chans_code = (cfg >> 11) & 0x03
    if chans_code == 0: nch = 2
    elif chans_code == 1: nch = 4
    elif chans_code == 2: nch = 8
    else: nch = 16

    res24 = ((cfg >> 7) & 1) == 1
    bps = 3 if res24 else 2
    hdr = 6 if res24 else 4  # (magic + timestamp sizes in your firmware)
    return nch, bps, hdr, res24

def expected_frame_len(nch: int, bps: int, hdr: int, blocks: int) -> int:
    # Your firmware: header + (numberOfBlocks * nch * bytesPerSample)
    return hdr + blocks * nch * bps

# ------------------------- Frame parser with buffering -------------------------
class FrameParser:
    def __init__(self, nch: int, bps: int, hdr: int, res24: bool, blocks: int):
        self.nch, self.bps, self.hdr, self.res24, self.blocks = nch, bps, hdr, res24, blocks
        self.buf = bytearray()
        self.frame_len = expected_frame_len(nch, bps, hdr, blocks)

    def feed(self, data: bytes):
        self.buf.extend(data)

    def _find_header(self, start: int) -> int:
        magic = MAG24 if self.res24 else MAG16
        return self.buf.find(magic, start)

    def _parse_one(self, j: int) -> Optional[Tuple[int, List[List[int]], int]]:
        """Parse frame at offset j. Returns (timestamp, samples[blocks][nch], next_index) or None."""
        n = len(self.buf)
        if j < 0 or j + self.hdr > n:
            return None
        # Timestamp
        if self.res24:
            # magic (3) + ts(3, BE)
            ts = (self.buf[j+3] << 16) | (self.buf[j+4] << 8) | self.buf[j+5]
            payload_start = j + 6
        else:
            # magic (2) + ts(2, BE)
            ts = (self.buf[j+2] << 8) | self.buf[j+3]
            payload_start = j + 4

        # Need a full frame
        if j + self.frame_len > n:
            return None

        payload = self.buf[payload_start : payload_start + (self.blocks * self.nch * self.bps)]
        samples = []
        k = 0
        for _ in range(self.blocks):
            row = []
            for _ch in range(self.nch):
                raw = payload[k : k + self.bps]
                if self.bps == 3:
                    val = int.from_bytes(raw, "big", signed=True)  # 24-bit signed
                else:
                    # 16-bit is packed as unsigned then converted back to signed in firmware logic
                    v = int.from_bytes(raw, "big", signed=False)
                    if v & 0x8000:
                        v -= 1 << 16
                    val = v
                row.append(val)
                k += self.bps
            samples.append(row)

        return ts, samples, j + self.frame_len

    def frames(self) -> List[Tuple[int, List[List[int]]]]:
        """Return list of complete frames parsed from buffer; buffer is trimmed."""
        out = []
        i = 0
        n = len(self.buf)
        while True:
            j = self._find_header(i)
            if j == -1:
                break
            parsed = self._parse_one(j)
            if not parsed:
                break  # need more data
            ts, samples, nxt = parsed
            out.append((ts, samples))
            i = nxt
        if i > 0:
            del self.buf[:i]
        return out

# ------------------------------------ CSV ------------------------------------
class CSVWriter:
    def __init__(self, path: str, nch: int):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["timestamp"] + [f"ch{i+1}" for i in range(nch)])
        self.rows = 0
    def write_samples(self, ts: int, block_rows: List[List[int]]):
        for row in block_rows:
            self.w.writerow([ts] + row)
            self.rows += 1
    def close(self):
        try: self.f.close()
        except: pass

# ---------------------------------- Main app ----------------------------------
async def main():
    ap = argparse.ArgumentParser(description="Auto-adaptive EMG BLE client (variable ch/res, handles fragmentation).")
    ap.add_argument("--addr", help="BLE address/UUID (skip scan)")
    ap.add_argument("--name", default=DEVICE_NAME_DEFAULT, help="Device name to scan")
    ap.add_argument("--config", default="0x00C3", help="16-bit config in hex (e.g. 0x00C3)")
    ap.add_argument("--blocks", type=int, default=10, help="numberOfBlocks per frame (firmware default=10)")
    ap.add_argument("--csv", default="", help="optional CSV output path")
    ap.add_argument("--peek", type=int, default=25, help="print every N frames (0=off)")
    args = ap.parse_args()

    cfg = int(args.config, 16) & 0xFFFF
    nch, bps, hdr, res24 = decode_config(cfg)
    frame_len = expected_frame_len(nch, bps, hdr, args.blocks)
    print(f"Config 0x{cfg:04X} → nch={nch}, bps={bps}, header={hdr}B, blocks={args.blocks}, expected frame={frame_len}B")

    # CSV
    writer = None
    if args.csv:
        path = args.csv
        writer = CSVWriter(path, nch)
        print(f"Writing CSV → {path}")

    # Resolve device
    if args.addr:
        address = args.addr
        print("Using address:", address)
    else:
        print(f"Scanning for {args.name} …")
        devs = await BleakScanner.discover(timeout=5.0)
        d = next((x for x in devs if (x.name or "") == args.name), None)
        if not d:
            print("Device not found. Seen:")
            for x in devs: print(" -", x.address, x.name)
            return
        address = d.address
        print("Found", address)

    parser = FrameParser(nch, bps, hdr, res24, args.blocks)
    frame_cnt = 0

    def on_notify(_, data: bytes):
        nonlocal frame_cnt, writer
        parser.feed(data)
        for ts, blocks in parser.frames():
            frame_cnt += 1
            if writer:
                writer.write_samples(ts, blocks)
            if args.peek and (frame_cnt % args.peek == 0):
                print(f"[frame {frame_cnt:06d}] ts={ts} len={len(blocks)} x {nch}ch"
                      f" (notif {len(data)}B, frame {parser.frame_len}B)"

                )

    # Send config (little-endian)
    cfg_le = bytes([cfg & 0xFF, (cfg >> 8) & 0xFF])

    try:
        async with BleakClient(address, timeout=20.0) as client:
            print("Connected:", client.is_connected)
            await client.start_notify(TX_UUID, on_notify)
            await client.write_gatt_char(RX_UUID, cfg_le, response=True)
            print(f"Sent config 0x{cfg:04X} (LE {cfg_le.hex()}). Streaming… Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(1.0)
            except KeyboardInterrupt:
                pass
            finally:
                try: await client.stop_notify(TX_UUID)
                except: pass
    finally:
        if writer:
            writer.close()
            print(f"Saved {writer.rows} rows to {args.csv}")

if __name__ == "__main__":
    asyncio.run(main())
