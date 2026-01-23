"""
Device Interface for Bracelet-EMG (BLE)

This module implements the PC-side interface for controlling and acquiring
biosignal data from the Bracelet-EMG firmware over Bluetooth Low Energy (BLE).

Responsibilities:
- Discover and connect to the Bracelet-EMG device via BLE
- Encode and transmit configuration commands
- Receive, frame, and decode streamed EMG data
- Persist decoded samples to CSV
- Provide a Qt-based GUI for user interaction
- Live plot (matplotlib embedded in PySide6)

Technologies:
- BLE: bleak
- GUI: PySide6 + qasync
- Async I/O: asyncio
- Data framing aligned with firmware protocol

This module is the authoritative reference implementation of the
Bracelet-EMG communication protocol on the PC side.
"""
import sys, asyncio, csv, datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Tuple

from PySide6 import QtCore
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGroupBox, QLabel, QComboBox,
    QPushButton, QRadioButton, QGridLayout, QVBoxLayout, QHBoxLayout
)
from qasync import QEventLoop, asyncSlot
from bleak import BleakScanner, BleakClient

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ---------------- Utils ----------------
def adc_to_uv(raw: int, res24: bool, gain: int, vref: float = 4.5) -> float:
    """Convert ADC codes to microvolts."""
    bits = 24 if res24 else 16
    lsb = (2 * vref) / (gain * (2**bits - 1))  # V / code
    return raw * lsb * 1e6


# ---------------- Live EMG plot ----------------
class EMGLivePlot(QWidget):
    """
    Live multi-channel EMG plot (stacked subplots).
    Expects input blocks already in microvolts (µV).
    """

    def __init__(self, fs: int, nch: int, seconds: float):
        super().__init__()
        self.fs = fs
        self.nch = nch
        self.seconds = seconds

        self.N = int(fs * seconds)
        self.t = np.arange(self.N) / fs

        # Plot decimation target ~500 Hz for smooth live plotting
        self.decim = max(1, fs // 500)
        self._counter = 0

        self.buffers = [np.zeros(self.N, dtype=float) for _ in range(nch)]
        self.dc = np.zeros(nch, dtype=float)

        self.fig = Figure(figsize=(10, 2.0 * nch))
        self.canvas = FigureCanvasQTAgg(self.fig)

        self.axes: List = []
        self.lines: List = []

        for ch in range(nch):
            ax = self.fig.add_subplot(nch, 1, ch + 1, sharex=self.axes[0] if ch else None)
            line, = ax.plot(self.t, self.buffers[ch], lw=0.8)

            ax.set_ylabel(f"ch{ch+1} (µV)")
            ax.grid(True)

            # Start with a sensible EMG range; user can tune later
            ax.set_ylim(-2000, 2000)

            self.axes.append(ax)
            self.lines.append(line)

        self.axes[-1].set_xlabel("Time (s)")
        self.fig.tight_layout()

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def push_blocks_uv(self, blocks_uv: List[List[float]]):
        """
        blocks_uv: List of rows, each row is [ch1_uv, ch2_uv, ...]
        """
        for row in blocks_uv:
            self._counter += 1
            if self._counter % self.decim != 0:
                continue

            for ch in range(self.nch):
                val = float(row[ch])

                # DC removal (slow IIR), keep consistent with offline if you also remove there
                self.dc[ch] = 0.999 * self.dc[ch] + 0.001 * val
                val -= self.dc[ch]

                self.buffers[ch][:-1] = self.buffers[ch][1:]
                self.buffers[ch][-1] = val

        for ch in range(self.nch):
            self.lines[ch].set_ydata(self.buffers[ch])

        self.canvas.draw_idle()


# ---------------- BLE UUIDs (must match firmware) ----------------
SVC_UUID = "12345678-1234-1234-1234-1234567890ab"
TX_UUID  = "12345678-1234-1234-1234-1234567890ac"
RX_UUID  = "12345678-1234-1234-1234-1234567890ad"

MAG24 = b"\x00\x62\x0f"
MAG16 = b"\x62\x0f"


def decode_config(cfg: int) -> Tuple[int, int, int, bool]:
    """
    Decode a 16-bit configuration command.

    Extracts channel count, bytes-per-sample, header length,
    and resolution mode from the packed configuration word.

    :param cfg: 16-bit configuration command
    :return: (num_channels, bytes_per_sample, header_size, is_24bit)
    """
    chans_code = (cfg >> 9) & 0x03
    nch = [2, 4, 8, 16][chans_code]
    res24 = ((cfg >> 6) & 1) == 1
    bps = 3 if res24 else 2
    hdr = 6 if res24 else 4
    return nch, bps, hdr, res24


def expected_frame_len(nch: int, bps: int, hdr: int, blocks: int) -> int:
    """
    Compute the expected frame length in bytes.

    :param nch: Number of channels
    :param bps: Bytes per sample
    :param hdr: Header size in bytes
    :param blocks: Number of blocks per frame
    :return: Total frame length in bytes
    """
    return hdr + blocks * nch * bps


class FrameParser:
    """
    Incremental decoder for Bracelet-EMG data frames.

    This class reconstructs full frames from fragmented BLE notifications.
    It handles:
    - Frame alignment using magic numbers
    - Timestamp extraction and rollover handling
    - Signed sample decoding (16-bit / 24-bit)
    - Block-wise sample grouping

    Frames are returned as:
        (timestamp, samples)

    where samples is a list of blocks, each block being a list of channel values.
    """
    def __init__(self, nch: int, bps: int, hdr: int, res24: bool, blocks: int):
        """
        Initialize a frame parser.

        :param nch: Number of channels
        :param bps: Bytes per sample
        :param hdr: Header size in bytes
        :param res24: True for 24-bit resolution
        :param blocks: Number of blocks per frame
        """
        self.nch, self.bps, self.hdr, self.res24, self.blocks = nch, bps, hdr, res24, blocks
        self.buf = bytearray()
        self.frame_len = expected_frame_len(nch, bps, hdr, blocks)

    def feed(self, data: bytes):
        """
        Feed raw BLE notification data into the parser.

        Data may contain partial frames, multiple frames,
        or frame fragments.
        """
        self.buf.extend(data)

    def _find_header(self, start: int) -> int:
        magic = MAG24 if self.res24 else MAG16
        return self.buf.find(magic, start)

    def _parse_one(self, j: int):
        n = len(self.buf)
        if j < 0 or j + self.hdr > n:
            return None

        # Timestamp
        if self.res24:
            ts = (self.buf[j+3] << 16) | (self.buf[j+4] << 8) | self.buf[j+5]
            payload_start = j + 6
        else:
            ts = (self.buf[j+2] << 8) | self.buf[j+3]
            payload_start = j + 4

        # Timestamp rollover
        if hasattr(self, "_last_ts"):
            if ts < self._last_ts:
                self._ts_base += (1 << (24 if self.res24 else 16))
        else:
            self._ts_base = 0
        self._last_ts = ts
        ts += self._ts_base

        if j + self.frame_len > n:
            return None

        payload = self.buf[payload_start: payload_start + (self.blocks*self.nch*self.bps)]
        samples = []
        k = 0

        for _ in range(self.blocks):
            row = []
            for _ in range(self.nch):
                raw = payload[k:k+self.bps]

                if self.bps == 3:  # 24-bit unsigned packed
                    v = int.from_bytes(raw, "big", signed=False)
                    if v & 0x800000:
                        v -= 1 << 24
                else:              # 16-bit unsigned packed
                    v = int.from_bytes(raw, "big", signed=False)
                    if v & 0x8000:
                        v -= 1 << 16

                row.append(v)
                k += self.bps
            samples.append(row)

        return ts, samples, j + self.frame_len

    def frames(self) -> List[Tuple[int, List[List[int]]]]:
        """
        Extract all complete frames currently available in the buffer.

        :return: List of decoded frames (timestamp, samples)
        """

        out = []
        i = 0
        magic = MAG24 if self.res24 else MAG16

        while True:
            j = self.buf.find(magic, i)
            if j == -1:
                break

            parsed = self._parse_one(j)
            if not parsed:
                break

            ts, samples, nxt = parsed

            # --- RESYNC CHECK ---
            # If there isn't another header exactly where we expect,
            # we might be misaligned (dropped bytes). In that case,
            # shift by 1 and search again.
            if nxt + len(magic) <= len(self.buf):
                if self.buf[nxt:nxt+len(magic)] != magic:
                    i = j + 1
                    continue

            out.append((ts, samples))
            i = nxt

        if i > 0:
            del self.buf[:i]
        return out

# ---------------- CSV writer ----------------
class CSVWriter:
    """
    Streaming CSV writer for EMG samples.

    Writes decoded EMG samples incrementally to disk,
    flushing periodically to reduce data loss risk.
    """
    def __init__(self, path: str, nch: int, flush_every: int = 200):
        self.f = open(path, "w", newline="", buffering=1)
        self.w = csv.writer(self.f)
        self.w.writerow(["timestamp"] + [f"ch{i+1}" for i in range(nch)])
        self.rows = 0
        self.flush_every = flush_every

    def write_samples(self, ts: int, block_rows: List[List[float]]):
        """
        Write a block of samples to CSV.

        :param ts: Timestamp associated with the block
        :param block_rows: List of per-channel sample rows
        """
        for row in block_rows:
            self.w.writerow([ts] + row)
            self.rows += 1
            if self.rows % self.flush_every == 0:
                self.f.flush()

    def close(self):
        try:
            self.f.flush()
        finally:
            try:
                self.f.close()
            except Exception:
                pass


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
    """
    Build a 16-bit configuration command.

    Encodes UI parameters into the firmware-defined
    configuration word format.

    :param p: UI parameters
    :param transmission_on: Enable or disable streaming
    :param set_config: Apply configuration immediately
    :return: Packed configuration word
    """
    cmd = 0
    cmd |= (1 << 13)
    cmd |= (p.fs_code & 0x03) << 11
    cmd |= (p.nch_code & 0x03) << 9
    cmd |= (p.mode_code & 0x03) << 7
    cmd |= (1 if p.res_24 else 0) << 6
    cmd |= (1 if p.hpf_on else 0) << 5
    # cmd |= (0) << 5  # HPF off
    cmd |= (p.gain_code & 0x07) << 2
    cmd |= (1 if set_config else 0) << 1
    cmd |= (1 if transmission_on else 0)
    return cmd & 0x3FFF


# ---------------- BLE Controller ----------------
class BLEController(QtCore.QObject):
    """
    Asynchronous BLE controller for Bracelet-EMG.

    Manages:
    - Device discovery and connection
    - Configuration transmission
    - Streaming lifecycle
    - Frame decoding and CSV logging

    Emits Qt signals for UI updates.
    """
    status = QtCore.Signal(str)
    streaming_changed = QtCore.Signal(bool)
    frames_progress = QtCore.Signal(int)

    # emits blocks already scaled to microvolts (µV)
    samples_ready_uv = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.client: Optional[BleakClient] = None
        self.parser: Optional[FrameParser] = None
        self.writer: Optional[CSVWriter] = None
        self.nch = 2
        self.blocks = 10
        self._frame_cnt = 0
        self.res24 = True
        self.gain = 1

    async def start(self, address: Optional[str], cfg: int, blocks: int, csv_path: Optional[str], gain: int):
        """
        Connect to the device, apply configuration, and start streaming.

        :param address: BLE MAC address or None to auto-scan
        :param cfg: Configuration word
        :param blocks: Number of blocks per frame
        :param csv_path: Optional CSV output path
        """
        self.nch, bps, hdr, self.res24 = decode_config(cfg)
        self.gain = int(gain)
        self.blocks = blocks
        self.parser = FrameParser(self.nch, bps, hdr, self.res24, blocks)
        self._frame_cnt = 0

        if csv_path:
            self.writer = CSVWriter(csv_path, self.nch)
            self.status.emit(f"CSV → {csv_path}")
        else:
            self.writer = None

        # If address is empty, scan for our bracelet
        if not address or (isinstance(address, str) and address.startswith("No devices")):
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
        """
        Stop streaming and disconnect from the device.

        Optionally sends a configuration command to disable transmission.
        """
        if self.client and cfg_off is not None:
            try:
                cfg_le = bytes([cfg_off & 0xFF, (cfg_off >> 8) & 0xFF])
                await self.client.write_gatt_char(RX_UUID, cfg_le, response=True)
                self.status.emit(f"Configured 0x{cfg_off:04X}; stopped.")
            except Exception:
                pass
        try:
            if self.client:
                try:
                    await self.client.stop_notify(TX_UUID)
                except Exception:
                    pass
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
            await asyncio.sleep(0.2)
        self.client = None
        if self.writer:
            self.writer.close()
            self.writer = None
        self.parser = None

    def _on_notify(self, _handle, data: bytes):
        """
        BLE notification handler.

        Receives raw BLE data, feeds it into the frame parser,
        and forwards decoded samples to the CSV writer.
        """
        if not self.parser:
            return

        self.parser.feed(data)

        for ts, blocks in self.parser.frames():
            self._frame_cnt += 1

            # Convert to µV ONCE here -> plot + CSV both use the same units
            blocks_uv = [
                [adc_to_uv(v, self.res24, self.gain) for v in row]
                for row in blocks
            ]

            self.samples_ready_uv.emit(blocks_uv)

            if self.writer:
                self.writer.write_samples(ts, blocks_uv)

            if (self._frame_cnt % 25) == 0:
                self.status.emit(
                    f"[frame {self._frame_cnt:06d}] ts={ts} len={len(blocks)} x {self.nch}ch"
                )
            self.frames_progress.emit(self._frame_cnt)


# ---------------- UI ----------------
class MainWindow(QMainWindow):
    """
    Main GUI window for Bracelet-EMG device control.

    Provides controls for:
    - BLE device selection
    - Acquisition parameters
    - Streaming start/stop
    - Real-time status updates
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("N-Switch Bracelet (BLE)")
        self.resize(900, 700)

        root = QWidget(self)
        self.setCentralWidget(root)
        self.v = QVBoxLayout(root)

        # Connection
        self.grp_conn = QGroupBox("Connection parameters")
        g = QGridLayout(self.grp_conn)
        g.addWidget(QLabel("Device"), 0, 0)
        self.device_name = QComboBox()
        g.addWidget(self.device_name, 0, 1)
        self.v.addWidget(self.grp_conn)

        # Acquisition
        self.grp_acq = QGroupBox("Acquisition Parameters")
        g = QGridLayout(self.grp_acq)
        g.addWidget(QLabel("Sampling Frequency"), 0, 0)
        self.cb_fs = QComboBox(); self.cb_fs.addItems(["500", "1000", "2000", "4000"]); self.cb_fs.setCurrentIndex(2)
        g.addWidget(self.cb_fs, 0, 1)
        g.addWidget(QLabel("Channels"), 1, 0)
        self.cb_nch = QComboBox(); self.cb_nch.addItems(["2", "4", "8", "16"]); self.cb_nch.setCurrentIndex(1)
        g.addWidget(self.cb_nch, 1, 1)
        self.v.addWidget(self.grp_acq)

        # Input
        self.grp_in = QGroupBox("Input Parameters")
        g = QGridLayout(self.grp_in)
        self.rb_16 = QRadioButton("16 Bit")
        self.rb_24 = QRadioButton("24 Bit"); self.rb_24.setChecked(True)
        g.addWidget(QLabel("Resolution"), 0, 0)
        rr = QHBoxLayout(); rr.addWidget(self.rb_16); rr.addWidget(self.rb_24)
        g.addLayout(rr, 0, 1)

        g.addWidget(QLabel("Mode"), 1, 0)
        self.cb_mode = QComboBox(); self.cb_mode.addItems(["MONOPOLAR","BIPOLAR","IMPEDANCE","TEST"]); self.cb_mode.setCurrentIndex(1)
        g.addWidget(self.cb_mode, 1, 1)

        g.addWidget(QLabel("Gain"), 2, 0)
        self.cb_gain = QComboBox(); self.cb_gain.addItems(["1", "2", "4", "6", "8", "12", "24"]); self.cb_gain.setCurrentIndex(4)
        g.addWidget(self.cb_gain, 2, 1)
        self.v.addWidget(self.grp_in)

        # Buttons
        row = QHBoxLayout()
        self.btn_stream = QPushButton("Stream")
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_stream); row.addWidget(self.btn_stop)
        self.v.addLayout(row)

        # Status
        self.lbl_status = QLabel("Idle")
        self.v.addWidget(self.lbl_status)

        # Live plot placeholder
        self.live_plot: Optional[EMGLivePlot] = None

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

    def _remove_live_plot(self):
        if self.live_plot is None:
            return
        try:
            self.ctrl.samples_ready_uv.disconnect(self.live_plot.push_blocks_uv)
        except Exception:
            pass
        self.live_plot.setParent(None)
        self.live_plot.deleteLater()
        self.live_plot = None

    def _on_streaming_changed(self, s: bool):
        self._disable_controls(s)
        self.lbl_status.setText("Streaming…" if s else "Stopped")
        if not s:
            # optional: keep plot visible after stop, or remove it:
            # self._remove_live_plot()
            pass

    @asyncSlot()
    async def _on_stream_clicked(self):
        # Clean old plot / old connections
        self._remove_live_plot()

        p = self._params_from_ui()
        cfg_on = build_cfg_bits(p, transmission_on=True, set_config=True)

        idx = self.device_name.currentIndex()
        address = self.device_name.itemData(idx)

        fs = int(self.cb_fs.currentText())
        nch = int(self.cb_nch.currentText())
        gain = int(self.cb_gain.currentText())

        gain_code = self.cb_gain.currentIndex()
        gain_value = [1,2,4,6,8,12,24][gain_code]

        # Create plot using actual UI values
        self.live_plot = EMGLivePlot(fs=fs, nch=nch, seconds=2.0)
        self.v.addWidget(self.live_plot)

        # Connect BLE -> plot (µV)
        self.ctrl.samples_ready_uv.connect(self.live_plot.push_blocks_uv)

        self.lbl_status.setText("Starting…")
        await self.ctrl.start(address, cfg_on, self.blocks, self._csv_path(), gain_value)

    @asyncSlot()
    async def _on_stop_clicked(self):
        p = self._params_from_ui()
        cfg_off = build_cfg_bits(p, transmission_on=False, set_config=True)
        await self.ctrl.stop(cfg_off)
        # Keep plot or remove; your choice:
        # self._remove_live_plot()

    def closeEvent(self, e):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.ctrl.stop())
        except RuntimeError:
            pass
        super().closeEvent(e)


def main():
    """
    Application entry point.

    Initializes Qt, asyncio event loop integration,
    and launches the main GUI window.
    """
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    win = MainWindow()
    win.show()
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
