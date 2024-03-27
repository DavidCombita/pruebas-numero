from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QWidget, QLineEdit, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from result import ResultWindow
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pruebas Estadísticas para Números Pseudoaleatorios")
        self.setGeometry(100, 100, 800, 400)

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.file_path_edit = QLineEdit()
        self.layout.addWidget(self.file_path_edit)

        self.load_file_button = QPushButton("Cargar Archivo")
        self.load_file_button.clicked.connect(self.load_file)
        self.layout.addWidget(self.load_file_button)

        self.create_button("Prueba de Medias", self.run_test_medias)
        self.create_button("Prueba de Varianza", self.run_test_varianza)
        self.create_button("Prueba KS", self.run_test_ks)
        self.create_button("Prueba Chi2", self.run_test_chi2)
        self.create_button("Prueba de Póker", self.run_test_poker)

    def create_button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        self.layout.addWidget(button)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo", "", "Archivos de Texto (*.num)")
        if file_path:
            self.file_path_edit.setText(file_path)

    def cargar_numeros(self):
        file_path = self.file_path_edit.text()
        with open(file_path, 'r') as file:
            data = file.read()
        numeros = data.split('#')
        numeros = [numero for numero in numeros if numero]  # Filtrar cadenas vacías
        return np.array([float(numero) for numero in numeros])

    def run_test_medias(self):
        numeros = self.cargar_numeros()
        p_valor = self.prueba_medias(numeros)
        self.show_result(p_valor, "Prueba de Medias", p_valor >= 0.05, numeros)

    def run_test_varianza(self):
        numeros = self.cargar_numeros()
        p_valor = self.prueba_varianza(numeros)
        self.show_result(p_valor,"Prueba de Varianza", p_valor >= 0.05, numeros)

    def run_test_ks(self):
        numeros = self.cargar_numeros()
        p_valor = self.prueba_ks(numeros)
        self.show_result(p_valor,"Prueba KS", p_valor >= 0.05, numeros)

    def run_test_chi2(self):
        numeros = self.cargar_numeros()
        p_valor = self.prueba_chi2(numeros)
        self.show_result(p_valor, "Prueba Chi2", p_valor >= 0.05, numeros)

    def run_test_poker(self):
        numeros = self.cargar_numeros()
        p_valor = self.prueba_poker(numeros)
        self.show_result(p_valor,"Prueba de Póker", p_valor >= 0.05,  numeros)

    def show_result(self, p_valor, prueba, passed, numeros):
        resultado = "Pasa" if passed else "No pasa"
        result_text = f"Resultado de la {prueba}: {resultado} \n"

        fig, ax = plt.subplots()
        if prueba == "Prueba de Medias":
            ax.hist(numeros, bins=10, edgecolor='black')
            ax.set_title('Histograma de Números Pseudoaleatorios')
            media_observada = np.mean(numeros)
            media_esperada = 0.5  # Para una distribución uniforme
            n = len(numeros)
            z = (media_observada - media_esperada) / (1 / (12 * n)**0.5)
            p_valor_media = 2 * (1 - stats.norm.cdf(abs(z)))
            result_text += f"Media observada: {media_observada:.4f}\n"
            result_text += f"Valor Z: {z:.4f}\n"
            result_text += f"Valor crítico: {stats.norm.ppf(1 - p_valor / 2):.4f}\n"
            result_text += f"Valor media: {p_valor_media:.4f}"
        elif prueba == "Prueba de Varianza":
            ax.hist(numeros, bins=10, edgecolor='black')
            varianza_observada = np.var(numeros)
            ax.axhline(y=varianza_observada, color='r', linestyle='-')
            ax.set_title('Histograma de Números Pseudoaleatorios y Varianza Observada')
            n = len(numeros)
            chi2 = (n - 1) * varianza_observada / 0.25
            p_valor_varianza = 1 - stats.chi2.cdf(chi2, n - 1)
            result_text += f"Varianza observada: {varianza_observada:.4f}\n"
            result_text += f"Valor Chi2: {chi2:.4f}\n"
            result_text += f"Valor crítico: {stats.chi2.ppf(1 - p_valor, n - 1):.4f}\n"
        elif prueba == "Prueba KS":
            _, p_valor_ks = stats.kstest(numeros, 'uniform')
            result_text += f"Valor de la prueba KS: {p_valor_ks:.4f}\n"
            result_text += f"Valor crítico: {stats.kstwobign.ppf(1 - p_valor, len(numeros)):.4f}\n"

            fig, ax = plt.subplots()
            ax.hist(numeros, bins=10, edgecolor='black', density=True, alpha=0.5, label='Datos')
            ax.set_title('Histograma de Números Pseudoaleatorios')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Densidad')
            # Generar datos para la distribución uniforme teórica
            x = np.linspace(0, 1, 1000)
            y = stats.uniform.cdf(x)
            ax.plot(x, y, 'r-', linewidth=2, label='Distribución Uniforme Teórica')
            ax.legend()
        elif prueba == "Prueba Chi2":
            frec_obs, _ = np.histogram(numeros, bins=10)
            n = len(numeros)
            frec_esp = np.full_like(frec_obs, n / 10)
            chi2, p_valor_chi2 = stats.chisquare(frec_obs, frec_esp)
            result_text += f"Valor de la prueba Chi2: {chi2:.4f}\n"
            result_text += f"Valor crítico: {stats.chi2.ppf(1 - p_valor, len(frec_obs) - 1):.4f}\n"

            fig, ax = plt.subplots()
            ax.bar(range(10), frec_obs, color='b', alpha=0.5, label='Frecuencia Observada')
            ax.plot(range(10), frec_esp, color='r', marker='o', linestyle='-', label='Frecuencia Esperada')
            ax.legend()
            ax.set_title('Gráfico de Frecuencias')
            ax.set_xlabel('Intervalo')
            ax.set_ylabel('Frecuencia')
        elif prueba == "Prueba de Póker":
            frecuencias = {}
            for numero in numeros:
                key = tuple(sorted(str(numero).replace('.', '')))
                if key in frecuencias:
                    frecuencias[key] += 1
                else:
                    frecuencias[key] = 1
            n = len(numeros)
            g = len(frecuencias)
            c = 0.005
            esperado = 0.001041666
            poker = (g * ((1 / (n - 4)) * sum([f**2 for f in frecuencias.values()])) - n) / (esperado * (g - 1))
            p_valor_poker = 1 - stats.chi2.cdf(poker, g - 1)
            result_text += f"Valor de la prueba de Póker: {poker:.4f}\n"
            result_text += f"Valor crítico: {stats.chi2.ppf(1 - p_valor, g - 1):.4f}\n"

            fig, ax = plt.subplots()
            ax.bar(range(len(frecuencias)), frecuencias.values(), color='b', alpha=0.5)
            ax.set_xticks(range(len(frecuencias)))
            ax.set_xticklabels(frecuencias.keys(), rotation=45)
            ax.set_title('Gráfico de Frecuencias Póker')
            ax.set_xlabel('Dígitos en orden ascendente')
            ax.set_ylabel('Frecuencia')
            
        fig.tight_layout()
        result_window = ResultWindow(result_text, fig)
        result_window.setWindowTitle("Resultado de la Prueba")
        result_window.exec_()



    def prueba_medias(self, numeros):
        media_observada = np.mean(numeros)
        media_esperada = 0.5
        n = len(numeros)
        z = (media_observada - media_esperada) / (1 / (12 * n) ** 0.5)
        p_valor = 2 * (1 - stats.norm.cdf(abs(z)))
        return p_valor

    def prueba_varianza(self, numeros):
        varianza_observada = np.var(numeros)
        varianza_esperada = 1 / 12
        n = len(numeros)
        chi2 = (n - 1) * varianza_observada / varianza_esperada
        p_valor = 1 - stats.chi2.cdf(chi2, n - 1)
        return p_valor

    def prueba_ks(self, numeros):
        _, p_valor = stats.kstest(numeros, 'uniform')
        return p_valor

    def prueba_chi2(self, numeros):
        frec_obs, _ = np.histogram(numeros, bins=10)
        n = len(numeros)
        frec_esp = np.full_like(frec_obs, n / 10)
        chi2, p_valor = stats.chisquare(frec_obs, frec_esp)
        return p_valor

    def prueba_poker(self, numeros):
        frecuencias = {}
        for numero in numeros:
            key = tuple(sorted(str(numero).replace('.', '')))
            if key in frecuencias:
                frecuencias[key] += 1
            else:
                frecuencias[key] = 1
        n = len(numeros)
        g = len(frecuencias)
        c = 0.005
        esperado = 0.001041666
        poker = (g * ((1 / (n - 4)) * sum([f**2 for f in frecuencias.values()])) - n) / (esperado * (g - 1))
        p_valor = 1 - stats.chi2.cdf(poker, g - 1)
        return p_valor

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
