from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ResultWindow(QDialog):
    def __init__(self, result_text, fig):
        super().__init__()
        layout = QVBoxLayout()
        label = QLabel(result_text)
        layout.addWidget(label)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self.setLayout(layout)