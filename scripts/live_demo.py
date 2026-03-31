import sys
import argparse
import queue
import threading
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

# Import from our project structure
from src.io.serial_receiver import SerialFrameReceiver
from src.dsp.preprocess import load_config, preprocess_frame

class LiveWaveformApp(QtWidgets.QMainWindow):
    def __init__(self, port, config_path):
        super().__init__()
        self.setWindowTitle("PQ Monitor - Live Oscilloscope")
        self.resize(1000, 600)
        
        # Load calibration data and expected samples
        self.cfg = load_config(config_path)
        self.expected_n = int(self.cfg["signal"]["samples_per_frame"])
        self.fs = int(self.cfg["signal"]["fs_hz"])
        
        # --- UI Setup ---
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # PyQtGraph widget
        pg.setConfigOptions(antialias=True)
        self.win = pg.GraphicsLayoutWidget()
        layout.addWidget(self.win)
        
        # Plot 1: Voltage
        self.p_v = self.win.addPlot(title="<span style='color: yellow; font-size: 14pt'>Voltage (V)</span>")
        self.p_v.showGrid(x=True, y=True, alpha=0.3)
        # self.p_v.setYRange(-400, 400) # Suitable range for 230V RMS (±325V peak) mains
        self.p_v.setLabel('left', 'Voltage', units='V')
        self.p_v.setLabel('bottom', 'Time', units='ms')
        self.curve_v = self.p_v.plot(pen=pg.mkPen('#FFD700', width=2))
        
        self.win.nextRow()
        
        # Plot 2: Current
        self.p_i = self.win.addPlot(title="<span style='color: cyan; font-size: 14pt'>Current (A)</span>")
        self.p_i.showGrid(x=True, y=True, alpha=0.3)
        # self.p_i.setYRange(-15, 15) # Dynamic scaling or fixed range depending on usual load 
        self.p_i.setLabel('left', 'Current', units='A')
        self.p_i.setLabel('bottom', 'Time', units='ms')
        self.curve_i = self.p_i.plot(pen=pg.mkPen('#00FFFF', width=2))
        
        # --- Data Queue & Receiver Thread ---
        # The receiver runs in a background thread and pushes processed frames into a queue.
        # This guarantees that our 60 FPS graphics UI never freezes if the serial USB drops or lags.
        self.data_queue = queue.Queue(maxsize=10)
        self.receiver = SerialFrameReceiver(port=port)
        
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        
        # --- UI Update Timer ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(33)  # Roughly 30 updates per second
        
    def _receive_loop(self):
        print(f"[*] Opening serial receiver on {self.receiver.port}...")
        self.receiver.open()
        
        while self.running:
            # Block until frame successfully parsed, with timeouts robustly handled
            frame = self.receiver.read_frame(frame_timeout=0.5)
            
            if frame is not None:
                try:
                    # Subtract DC offset, scale raw ADC counts mapping to real AMPS and VOLTS
                    processed = preprocess_frame(frame.v_raw, frame.i_raw, self.cfg, expected_n=self.expected_n)
                    
                    # Backpressure drop policy: if queue is full (UI is rendering too slow), drop oldest frame
                    if self.data_queue.full():
                        try:
                            self.data_queue.get_nowait()
                        except queue.Empty:
                            pass
                            
                    self.data_queue.put(processed)
                    
                    # HOTFIX PRINT: Show exact raw 0-4095 ADC values straight from the Teensy firmware block
                    print(f"RAW A0: {frame.v_raw[0]}    |    RAW A10: {frame.i_raw[0]}")
                except Exception as e:
                    print(f"[!] Preprocess error: {e}")
                    
    def update_plots(self):
        try:
            latest = None
            # Drain queue cleanly to always render the "freshest" snapshot
            while not self.data_queue.empty():
                latest = self.data_queue.get_nowait()
                
            if latest is not None:
                v_phys = latest["v_phys"]
                i_phys = latest["i_phys"]
                
                # Fast time axis creation for N=500 -> 0 ms to 100 ms bounds
                t = np.linspace(0, (len(v_phys) / self.fs) * 1000, len(v_phys))
                
                # Push vectors down to C++ Qt boundary over pyqtgraph wrapper
                self.curve_v.setData(t, v_phys)
                self.curve_i.setData(t, i_phys)
        except Exception:
            pass

    def closeEvent(self, event):
        print("[*] Shutting down...")
        self.running = False
        self.receiver.close()
        self.thread.join(timeout=1.0)
        event.accept()

def main():
    parser = argparse.ArgumentParser(description="Live Power Quality Form Visualizer")
    parser.add_argument("--port", required=True, help="Serial port of Teensy (e.g. /dev/ttyACM0)")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    
    # Optional styling for a dark "Oscilloscope" theme
    app.setStyle("Fusion")
    
    window = LiveWaveformApp(args.port, args.config)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
