import sys
import numpy as np, scipy, h5py, sklearn, torch
from PyQt5 import QtWidgets
import pyqtgraph as pg

print("Python   ", sys.version.split()[0], "|", sys.executable)
print("torch    ", torch.__version__, "| built for CUDA:", torch.version.cuda)
print("CUDA OK  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU      ", torch.cuda.get_device_name(0))
    x = torch.randn(1000, 1000, device="cuda")
    print("GPU test ", (x @ x).sum().item() != 0)
print("h5py     ", h5py.__version__, "| HDF5:", h5py.version.hdf5_version)
print("pyqtgraph", pg.__version__)

app = QtWidgets.QApplication(sys.argv)
win = pg.plot(np.sin(np.linspace(0, 20, 1000)), title="Setup OK – close this window")
app.exec_()