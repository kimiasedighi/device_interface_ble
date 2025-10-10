# import sys, asyncio, csv, datetime as dt
# from dataclasses import dataclass
# from typing import Optional, List, Tuple
# from PySide6 import QtWidgets, QtCore
# from PySide6.QtWidgets import (
#     QApplication, QMainWindow, QWidget, QGroupBox, QLabel, QComboBox,
#     QPushButton, QRadioButton, QGridLayout, QVBoxLayout, QHBoxLayout
# )
# from qasync import QEventLoop, asyncSlot
# from bleak import BleakScanner, BleakClient

# # ---------------- BLE UUIDs (must match firmware) ----------------
# SVC_UUID = "12345678-1234-1234-1234-1234567890ab"
# TX_UUID  = "12345678-1234-1234-1234-1234567890ac"   # notify
# RX_UUID  = "12345678-1234-1234-1234-1234567890ad"   # write

# # ---------------- Framing (matches your microcontroller) ----------------
# MAG24 = b"\x00\x62\x0f"   # magic(3) + ts(3, BE)
# MAG16 = b"\x62\x0f"       # magic(2) + ts(2, BE)

# def decode_config(cfg: int) -> Tuple[int, int, int, bool]:
#     chans_code = (cfg >> 9) & 0x03
#     nch = [2, 4, 8, 16][chans_code]

#     res24 = ((cfg >> 6) & 1) == 1
#     bps = 3 if res24 else 2
#     hdr = 6 if res24 else 4
#     return nch, bps, hdr, res24

# def expected_frame_len(nch: int, bps: int, hdr: int, blocks: int) -> int:
#     return hdr + blocks * nch * bps

# class FrameParser:
#     def __init__(self, nch: int, bps: int, hdr: int, res24: bool, blocks: int):
#         self.nch, self.bps, self.hdr, self.res24, self.blocks = nch, bps, hdr, res24, blocks
#         self.buf = bytearray()
#         self.frame_len = expected_frame_len(nch, bps, hdr, blocks)

#     def feed(self, data: bytes):
#         self.buf.extend(data)

#     def _find_header(self, start: int) -> int:
#         magic = MAG24 if self.res24 else MAG16
#         return self.buf.find(magic, start)

#     def _parse_one(self, j: int):
#         n = len(self.buf)
#         if j < 0 or j + self.hdr > n:
#             return None
#         if self.res24:
#             ts = (self.buf[j+3] << 16) | (self.buf[j+4] << 8) | self.buf[j+5]
#             payload_start = j + 6
#         else:
#             ts = (self.buf[j+2] << 8) | self.buf[j+3]
#             payload_start = j + 4

#         if j + self.frame_len > n:
#             return None

#         payload = self.buf[payload_start : payload_start + (self.blocks * self.nch * self.bps)]
#         samples: List[List[int]] = []
#         k = 0
#         for _ in range(self.blocks):
#             row = []
#             for _ch in range(self.nch):
#                 raw = payload[k : k + self.bps]
#                 if self.bps == 3:
#                     val = int.from_bytes(raw, "big", signed=True)
#                 else:
#                     v = int.from_bytes(raw, "big", signed=False)
#                     if v & 0x8000:
#                         v -= 1 << 16
#                     val = v
#                 row.append(val)
#                 k += self.bps
#             samples.append(row)
#         return ts, samples, j + self.frame_len

#     def frames(self) -> List[Tuple[int, List[List[int]]]]:
#         out = []
#         i = 0
#         while True:
#             j = self._find_header(i)
#             if j == -1:
#                 break
#             parsed = self._parse_one(j)
#             if not parsed:
#                 break
#             ts, samples, nxt = parsed
#             out.append((ts, samples))
#             i = nxt
#         if i > 0:
#             del self.buf[:i]
#         return out

# # ---------------- CSV writer ----------------
# class CSVWriter:
#     def __init__(self, path: str, nch: int, flush_every: int = 100):
#         self.f = open(path, "w", newline="", buffering=1)
#         self.w = csv.writer(self.f)
#         self.w.writerow(["timestamp"] + [f"ch{i+1}" for i in range(nch)])
#         self.rows = 0
#         self.flush_every = flush_every
#     def write_samples(self, ts: int, block_rows: List[List[int]]):
#         for row in block_rows:
#             self.w.writerow([ts] + row)
#             self.rows += 1
#             if self.rows % self.flush_every == 0:
#                 self.f.flush()
#     def close(self):
#         try:
#             self.f.flush()
#         finally:
#             try: self.f.close()
#             except: pass

# # ---------------- UI Parameters ----------------
# @dataclass
# class UIParams:
#     fs_code: int
#     nch_code: int
#     mode_code: int
#     res_24: bool
#     hpf_on: bool
#     gain_code: int

# def build_cfg_bits(p: UIParams, transmission_on: bool, set_config: bool = True) -> int:
#     cmd = 0
#     cmd |= (p.fs_code  & 0x03) << 11
#     cmd |= (p.nch_code & 0x03) << 9
#     cmd |= (p.mode_code & 0x03) << 7
#     cmd |= (1 if p.res_24 else 0) << 6
#     cmd |= (1 if p.hpf_on else 0) << 5
#     cmd |= (p.gain_code & 0x07) << 2
#     cmd |= (1 if set_config else 0) << 1
#     cmd |= (1 if transmission_on else 0)
#     return cmd & 0x3FFF

# # ---------------- Controller ----------------
# class BLEController(QtCore.QObject):
#     status = QtCore.Signal(str)
#     streaming_changed = QtCore.Signal(bool)
#     frames_progress = QtCore.Signal(int)

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.client: Optional[BleakClient] = None
#         self.parser: Optional[FrameParser] = None
#         self.writer: Optional[CSVWriter] = None
#         self.nch = 2
#         self.blocks = 10
#         self._frame_cnt = 0
#         self._address = None

#     async def start(self, address: str, cfg: int, blocks: int, csv_path: Optional[str]):
#         self.nch, bps, hdr, res24 = decode_config(cfg)
#         self.blocks = blocks
#         self.parser = FrameParser(self.nch, bps, hdr, res24, blocks)
#         self._frame_cnt = 0
#         self._address = address

#         if csv_path:
#             self.writer = CSVWriter(csv_path, self.nch)
#             self.status.emit(f"CSV → {csv_path}")
#         else:
#             self.writer = None

#         self.status.emit(f"Connecting to {self._address}…")
#         self.client = BleakClient(self._address, timeout=20.0)
#         await self.client.__aenter__()
#         if not self.client.is_connected:
#             self.status.emit("Connect failed")
#             await self._cleanup()
#             return
#         self.status.emit("Connected")

#         await self.client.start_notify(TX_UUID, self._on_notify)

#         cfg_le = bytes([cfg & 0xFF, (cfg >> 8) & 0xFF])
#         await self.client.write_gatt_char(RX_UUID, cfg_le, response=True)
#         self.status.emit(f"Configured 0x{cfg:04X}; streaming…")
#         self.streaming_changed.emit(True)

#     async def stop(self, cfg_off: Optional[int] = None):
#         if self.client and cfg_off is not None:
#             try:
#                 cfg_le = bytes([cfg_off & 0xFF, (cfg_off >> 8) & 0xFF])
#                 await self.client.write_gatt_char(RX_UUID, cfg_le, response=True)
#                 self.status.emit(f"Configured 0x{cfg_off:04X}; stopped.")
#             except Exception:
#                 pass
#         try:
#             if self.client:
#                 try: await self.client.stop_notify(TX_UUID)
#                 except Exception: pass
#         finally:
#             await self._cleanup()
#             self.streaming_changed.emit(False)
#             if self.writer:
#                 self.status.emit(f"Saved {self.writer.rows} rows to {self.writer.f.name}")

#     async def _cleanup(self):
#         if self.client:
#             try:
#                 await self.client.__aexit__(None, None, None)
#             except Exception:
#                 pass
#         self.client = None
#         if self.writer:
#             self.writer.close()
#             self.writer = None
#         self.parser = None

#     def _on_notify(self, _handle, data: bytes):
#         if not self.parser:
#             return
#         self.parser.feed(data)
#         for ts, blocks in self.parser.frames():
#             self._frame_cnt += 1
#             if self.writer:
#                 self.writer.write_samples(ts, blocks)
#             if (self._frame_cnt % 25) == 0:
#                 self.status.emit(
#                     f"[frame {self._frame_cnt:06d}] ts={ts} len={len(blocks)} x {self.nch}ch"
#                 )
#             self.frames_progress.emit(self._frame_cnt)

# # ---------------- Main Window ----------------
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("N-Switch Bracelet (BLE)")
#         self.resize(520, 430)

#         root = QWidget(self)
#         self.setCentralWidget(root)
#         v = QVBoxLayout(root)

#         # Connection group
#         self.grp_conn = QGroupBox("Connection parameters")
#         g = QGridLayout(self.grp_conn)
#         g.addWidget(QLabel("Device"), 0, 0)
#         self.device_name = QComboBox()
#         g.addWidget(self.device_name, 0, 1)
#         v.addWidget(self.grp_conn)

#         # Acquisition group
#         self.grp_acq = QGroupBox("Acquisition Parameters")
#         g = QGridLayout(self.grp_acq)
#         g.addWidget(QLabel("Sampling Frequency"), 0, 0)
#         self.cb_fs = QComboBox(); self.cb_fs.addItems(["500", "1000", "2000", "4000"]); self.cb_fs.setCurrentIndex(0)
#         g.addWidget(self.cb_fs, 0, 1)
#         g.addWidget(QLabel("Number of Channels"), 1, 0)
#         self.cb_nch = QComboBox(); self.cb_nch.addItems(["2", "4", "8", "16"]); self.cb_nch.setCurrentIndex(0)
#         g.addWidget(self.cb_nch, 1, 1)
#         v.addWidget(self.grp_acq)

#         # Input group
#         self.grp_in = QGroupBox("Input Parameters")
#         g = QGridLayout(self.grp_in)
#         self.rb_16 = QRadioButton("16 Bit"); self.rb_24 = QRadioButton("24 Bit"); self.rb_24.setChecked(True)
#         g.addWidget(QLabel("Resolution"), 0, 0); rr = QHBoxLayout(); rr.addWidget(self.rb_16); rr.addWidget(self.rb_24); g.addLayout(rr, 0, 1)
#         g.addWidget(QLabel("Mode"), 1, 0)
#         self.cb_mode = QComboBox(); self.cb_mode.addItems(["MONOPOLAR","BIPOLAR","IMPEDANCE","TEST"])
#         g.addWidget(self.cb_mode, 1, 1)
#         g.addWidget(QLabel("Gain"), 2, 0)
#         self.cb_gain = QComboBox(); 
#         self.cb_gain.addItems(["6 (Default)", "1", "2", "3", "4", "8", "12"])
#         g.addWidget(self.cb_gain, 2, 1)
#         v.addWidget(self.grp_in)

#         # Buttons
#         row = QHBoxLayout()
#         self.btn_stream = QPushButton("Stream"); self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False)
#         row.addWidget(self.btn_stream); row.addWidget(self.btn_stop)
#         v.addLayout(row)

#         # Status
#         self.lbl_status = QLabel("Idle"); v.addWidget(self.lbl_status)

#         # Controller
#         self.ctrl = BLEController()
#         self.ctrl.status.connect(self.lbl_status.setText)
#         self.ctrl.streaming_changed.connect(self._on_streaming_changed)

#         # Events
#         self.btn_stream.clicked.connect(self._on_stream_clicked)
#         self.btn_stop.clicked.connect(self._on_stop_clicked)

#         # Defaults
#         self.blocks = 10

#         # Auto-refresh device list
#         QtCore.QTimer.singleShot(100, lambda: asyncio.create_task(self._refresh_devices()))


#     async def _refresh_devices(self):
#         self.device_name.clear()
#         self.device_name.addItem("Scanning…")
#         await asyncio.sleep(0)

#         devs = await BleakScanner.discover(timeout=5.0)
#         self.device_name.clear()
#         if not devs:
#             self.device_name.addItem("No devices found")
#             return

#         for d in devs:
#             label = f"{d.name or 'Unknown'} ({d.address})"
#             self.device_name.addItem(label, userData=d.address)

#     def _params_from_ui(self) -> UIParams:
#         return UIParams(
#             fs_code=self.cb_fs.currentIndex(),
#             nch_code=self.cb_nch.currentIndex(),
#             mode_code=self.cb_mode.currentIndex(),
#             res_24=self.rb_24.isChecked(),
#             hpf_on=True,
#             gain_code=self.cb_gain.currentIndex(),
#         )

#     def _csv_path(self) -> str:
#         ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
#         return f"emg_{ts}.csv"

#     def _disable_controls(self, disable: bool):
#         self.grp_conn.setEnabled(not disable)
#         self.grp_acq.setEnabled(not disable)
#         self.grp_in.setEnabled(not disable)
#         self.btn_stream.setEnabled(not disable)
#         self.btn_stop.setEnabled(disable)

#     def _on_streaming_changed(self, s: bool):
#         self._disable_controls(s)
#         self.lbl_status.setText("Streaming…" if s else "Stopped")

#     @asyncSlot()
#     async def _on_stream_clicked(self):
#         p = self._params_from_ui()
#         cfg_on = build_cfg_bits(p, transmission_on=True, set_config=True)

#         idx = self.device_name.currentIndex()
#         address = self.device_name.itemData(idx)

#         self.lbl_status.setText("Starting…")
#         await self.ctrl.start(address, cfg_on, self.blocks, self._csv_path())

#     @asyncSlot()
#     async def _on_stop_clicked(self):
#         p = self._params_from_ui()
#         cfg_off = build_cfg_bits(p, transmission_on=False, set_config=True)
#         await self.ctrl.stop(cfg_off)

#     def closeEvent(self, e):
#         try:
#             loop = asyncio.get_event_loop()
#             if loop.is_running():
#                 loop.create_task(self.ctrl.stop())
#         except RuntimeError:
#             pass
#         super().closeEvent(e)

# def main():
#     app = QApplication(sys.argv)
#     loop = QEventLoop(app)
#     asyncio.set_event_loop(loop)
#     win = MainWindow()
#     win.show()
#     with loop:
#         loop.run_forever()

# if __name__ == "__main__":
#     main()


import sys, asyncio, csv, datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Tuple
from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGroupBox, QLabel, QComboBox,
    QPushButton, QRadioButton, QGridLayout, QVBoxLayout, QHBoxLayout
)
from qasync import QEventLoop, asyncSlot
from bleak import BleakScanner, BleakClient

# ---------------- BLE UUIDs (must match firmware) ----------------
SVC_UUID = "12345678-1234-1234-1234-1234567890ab"
TX_UUID  = "12345678-1234-1234-1234-1234567890ac"
RX_UUID  = "12345678-1234-1234-1234-1234567890ad"

# ---------------- Framing ----------------
MAG24 = b"\x00\x62\x0f"
MAG16 = b"\x62\x0f"

def decode_config(cfg: int) -> Tuple[int, int, int, bool]:
    chans_code = (cfg >> 9) & 0x03
    nch = [2, 4, 8, 16][chans_code]
    res24 = ((cfg >> 6) & 1) == 1
    bps = 3 if res24 else 2
    hdr = 6 if res24 else 4
    return nch, bps, hdr, res24

def expected_frame_len(nch: int, bps: int, hdr: int, blocks: int) -> int:
    return hdr + blocks * nch * bps

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

    def _parse_one(self, j: int):
        n = len(self.buf)
        if j < 0 or j + self.hdr > n:
            return None
        if self.res24:
            ts = (self.buf[j+3] << 16) | (self.buf[j+4] << 8) | self.buf[j+5]
            payload_start = j + 6
        else:
            ts = (self.buf[j+2] << 8) | self.buf[j+3]
            payload_start = j + 4
        if j + self.frame_len > n:
            return None
        payload = self.buf[payload_start : payload_start + (self.blocks * self.nch * self.bps)]
        samples: List[List[int]] = []
        k = 0
        for _ in range(self.blocks):
            row = []
            for _ch in range(self.nch):
                raw = payload[k : k + self.bps]
                if self.bps == 3:
                    val = int.from_bytes(raw, "big", signed=True)
                else:
                    v = int.from_bytes(raw, "big", signed=False)
                    if v & 0x8000:
                        v -= 1 << 16
                    val = v
                row.append(val)
                k += self.bps
            samples.append(row)
        return ts, samples, j + self.frame_len

    def frames(self) -> List[Tuple[int, List[List[int]]]]:
        out = []
        i = 0
        while True:
            j = self._find_header(i)
            if j == -1:
                break
            parsed = self._parse_one(j)
            if not parsed:
                break
            ts, samples, nxt = parsed
            out.append((ts, samples))
            i = nxt
        if i > 0:
            del self.buf[:i]
        return out

# ---------------- CSV writer ----------------
class CSVWriter:
    def __init__(self, path: str, nch: int, flush_every: int = 100):
        self.f = open(path, "w", newline="", buffering=1)
        self.w = csv.writer(self.f)
        self.w.writerow(["timestamp"] + [f"ch{i+1}" for i in range(nch)])
        self.rows = 0
        self.flush_every = flush_every
    def write_samples(self, ts: int, block_rows: List[List[int]]):
        for row in block_rows:
            self.w.writerow([ts] + row)
            self.rows += 1
            if self.rows % self.flush_every == 0:
                self.f.flush()
    def close(self):
        try:
            self.f.flush()
        finally:
            try: self.f.close()
            except: pass

# ---------------- Config container ----------------
@dataclass
class UIParams:
    fs_code: int
    nch_code: int
    mode_code: int
    res_24: bool
    hpf_on: bool
    gain_code: int

def build_cfg_bits(p: UIParams, transmission_on: bool, set_config: bool = True) -> int:
    cmd = 0
    cmd |= (p.fs_code  & 0x03) << 11
    cmd |= (p.nch_code & 0x03) << 9
    cmd |= (p.mode_code & 0x03) << 7
    cmd |= (1 if p.res_24 else 0) << 6
    cmd |= (1 if p.hpf_on else 0) << 5
    cmd |= (p.gain_code & 0x07) << 2
    cmd |= (1 if set_config else 0) << 1
    cmd |= (1 if transmission_on else 0)
    return cmd & 0x3FFF

# ---------------- BLE Controller ----------------
class BLEController(QtCore.QObject):
    status = QtCore.Signal(str)
    streaming_changed = QtCore.Signal(bool)
    frames_progress = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.client: Optional[BleakClient] = None
        self.parser: Optional[FrameParser] = None
        self.writer: Optional[CSVWriter] = None
        self.nch = 2
        self.blocks = 10
        self._frame_cnt = 0

    async def start(self, address: Optional[str], cfg: int, blocks: int, csv_path: Optional[str]):
        self.nch, bps, hdr, res24 = decode_config(cfg)
        self.blocks = blocks
        self.parser = FrameParser(self.nch, bps, hdr, res24, blocks)
        self._frame_cnt = 0

        if csv_path:
            self.writer = CSVWriter(csv_path, self.nch)
            self.status.emit(f"CSV → {csv_path}")
        else:
            self.writer = None

        # If address is empty, scan for our bracelet
        if not address or address.startswith("No devices"):
            self.status.emit("Scanning for Bracelet-EMG…")
            dev = await BleakScanner.find_device_by_filter(
                lambda d, ad: (d.name and "bracelet" in d.name.lower()) or
                              (SVC_UUID.lower() in [u.lower() for u in (ad.service_uuids or [])]),
                timeout=10.0
            )
            if not dev:
                self.status.emit("Device not found.")
                return
            address = dev.address

        self.status.emit(f"Connecting to {address}…")
        try:
            self.client = BleakClient(address, timeout=20.0)
            await self.client.__aenter__()
            if not self.client.is_connected:
                raise Exception("Connect failed.")
        except Exception as e:
            self.status.emit(f"Connection failed: {e}")
            await self._cleanup()
            return

        self.status.emit("Connected.")
        await self.client.start_notify(TX_UUID, self._on_notify)

        cfg_le = bytes([cfg & 0xFF, (cfg >> 8) & 0xFF])
        await self.client.write_gatt_char(RX_UUID, cfg_le, response=True)
        self.status.emit(f"Configured 0x{cfg:04X}; streaming…")
        self.streaming_changed.emit(True)

    async def stop(self, cfg_off: Optional[int] = None):
        if self.client and cfg_off is not None:
            try:
                cfg_le = bytes([cfg_off & 0xFF, (cfg_off >> 8) & 0xFF])
                await self.client.write_gatt_char(RX_UUID, cfg_le, response=True)
                self.status.emit(f"Configured 0x{cfg_off:04X}; stopped.")
            except Exception:
                pass
        try:
            if self.client:
                try: await self.client.stop_notify(TX_UUID)
                except Exception: pass
        finally:
            await self._cleanup()
            self.streaming_changed.emit(False)
            if self.writer:
                self.status.emit(f"Saved {self.writer.rows} rows to {self.writer.f.name}")

    async def _cleanup(self):
        if self.client:
            try:
                await self.client.disconnect()
            except Exception:
                pass
            try:
                await self.client.__aexit__(None, None, None)
            except Exception:
                pass
            await asyncio.sleep(0.3)
        self.client = None
        if self.writer:
            self.writer.close()
            self.writer = None
        self.parser = None

    def _on_notify(self, _handle, data: bytes):
        if not self.parser:
            return
        self.parser.feed(data)
        for ts, blocks in self.parser.frames():
            self._frame_cnt += 1
            if self.writer:
                self.writer.write_samples(ts, blocks)
            if (self._frame_cnt % 25) == 0:
                self.status.emit(
                    f"[frame {self._frame_cnt:06d}] ts={ts} len={len(blocks)} x {self.nch}ch"
                )
            self.frames_progress.emit(self._frame_cnt)

# ---------------- UI ----------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("N-Switch Bracelet (BLE)")
        self.resize(520, 430)

        root = QWidget(self)
        self.setCentralWidget(root)
        v = QVBoxLayout(root)

        # Connection group
        self.grp_conn = QGroupBox("Connection parameters")
        g = QGridLayout(self.grp_conn)
        g.addWidget(QLabel("Device"), 0, 0)
        self.device_name = QComboBox()
        g.addWidget(self.device_name, 0, 1)
        v.addWidget(self.grp_conn)

        # Acquisition
        self.grp_acq = QGroupBox("Acquisition Parameters")
        g = QGridLayout(self.grp_acq)
        g.addWidget(QLabel("Sampling Frequency"), 0, 0)
        self.cb_fs = QComboBox(); self.cb_fs.addItems(["500", "1000", "2000", "4000"]); self.cb_fs.setCurrentIndex(0)
        g.addWidget(self.cb_fs, 0, 1)
        g.addWidget(QLabel("Channels"), 1, 0)
        self.cb_nch = QComboBox(); self.cb_nch.addItems(["2", "4", "8", "16"]); self.cb_nch.setCurrentIndex(0)
        g.addWidget(self.cb_nch, 1, 1)
        v.addWidget(self.grp_acq)

        # Input
        self.grp_in = QGroupBox("Input Parameters")
        g = QGridLayout(self.grp_in)
        self.rb_16 = QRadioButton("16 Bit"); self.rb_24 = QRadioButton("24 Bit"); self.rb_24.setChecked(True)
        g.addWidget(QLabel("Resolution"), 0, 0)
        rr = QHBoxLayout(); rr.addWidget(self.rb_16); rr.addWidget(self.rb_24)
        g.addLayout(rr, 0, 1)
        g.addWidget(QLabel("Mode"), 1, 0)
        self.cb_mode = QComboBox(); self.cb_mode.addItems(["MONOPOLAR","BIPOLAR","IMPEDANCE","TEST"])
        g.addWidget(self.cb_mode, 1, 1)
        g.addWidget(QLabel("Gain"), 2, 0)
        self.cb_gain = QComboBox(); self.cb_gain.addItems(["6 (Default)", "1", "2", "3", "4", "8", "12"])
        g.addWidget(self.cb_gain, 2, 1)
        v.addWidget(self.grp_in)

        # Buttons
        row = QHBoxLayout()
        self.btn_stream = QPushButton("Stream")
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_stream); row.addWidget(self.btn_stop)
        v.addLayout(row)

        # Status
        self.lbl_status = QLabel("Idle"); v.addWidget(self.lbl_status)

        # Controller
        self.ctrl = BLEController()
        self.ctrl.status.connect(self.lbl_status.setText)
        self.ctrl.streaming_changed.connect(self._on_streaming_changed)

        # Events
        self.btn_stream.clicked.connect(self._on_stream_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        self.blocks = 10
        QtCore.QTimer.singleShot(100, lambda: asyncio.create_task(self._refresh_devices()))

    async def _refresh_devices(self):
        self.device_name.clear()
        self.device_name.addItem("Scanning…")
        await asyncio.sleep(0)
        devs = await BleakScanner.discover(timeout=5.0)
        self.device_name.clear()
        if not devs:
            self.device_name.addItem("No devices found")
            return
        for d in devs:
            if not d.name:
                continue
            label = f"{d.name} ({d.address})"
            self.device_name.addItem(label, userData=d.address)

    def _params_from_ui(self) -> UIParams:
        return UIParams(
            fs_code=self.cb_fs.currentIndex(),
            nch_code=self.cb_nch.currentIndex(),
            mode_code=self.cb_mode.currentIndex(),
            res_24=self.rb_24.isChecked(),
            hpf_on=True,
            gain_code=self.cb_gain.currentIndex(),
        )

    def _csv_path(self) -> str:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"emg_{ts}.csv"

    def _disable_controls(self, disable: bool):
        self.grp_conn.setEnabled(not disable)
        self.grp_acq.setEnabled(not disable)
        self.grp_in.setEnabled(not disable)
        self.btn_stream.setEnabled(not disable)
        self.btn_stop.setEnabled(disable)

    def _on_streaming_changed(self, s: bool):
        self._disable_controls(s)
        self.lbl_status.setText("Streaming…" if s else "Stopped")

    @asyncSlot()
    async def _on_stream_clicked(self):
        p = self._params_from_ui()
        cfg_on = build_cfg_bits(p, transmission_on=True, set_config=True)
        idx = self.device_name.currentIndex()
        address = self.device_name.itemData(idx)
        self.lbl_status.setText("Starting…")
        await self.ctrl.start(address, cfg_on, self.blocks, self._csv_path())

    @asyncSlot()
    async def _on_stop_clicked(self):
        p = self._params_from_ui()
        cfg_off = build_cfg_bits(p, transmission_on=False, set_config=True)
        await self.ctrl.stop(cfg_off)

    def closeEvent(self, e):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.ctrl.stop())
        except RuntimeError:
            pass
        super().closeEvent(e)

def main():
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    win = MainWindow()
    win.show()
    with loop:
        loop.run_forever()

if __name__ == "__main__":
    main()
