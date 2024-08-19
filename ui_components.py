import os
from PyQt6 import QtWidgets, uic
from data_processor import DataProcessor
from plotter import PlotWindow
from analyzer import Analyzer
from utils import set_default_values, get_values_from_ui
from hyperparameters_window import HyperparametersWindow


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi(os.path.join("ui", "app.ui"), self)
        self.setWindowTitle("Spectral FLIM Analysis")
        self.data_processor = DataProcessor()
        self.analyzer = Analyzer()
        self.setup_ui()
        self.setup_connections()
        self.set_default_values()

    def setup_ui(self):
        self.ui.progressBar.setVisible(False)
        self.ui.plot_.setEnabled(False)
        self.ui.save_results.setEnabled(False)

    def setup_connections(self):
        self.ui.load_data.clicked.connect(self.browse_file)
        self.ui.run_.clicked.connect(self.run_analysis)
        self.ui.HyperParameters_.triggered.connect(self.open_hyperparameters)
        self.ui.plot_.clicked.connect(self.show_plot)
        self.ui.save_results.clicked.connect(self.export_results)

    def set_default_values(self):
        default_values = set_default_values()
        self.ui.num_species.setText(str(default_values["num_species"]))
        self.ui.t_inter_p.setText(str(default_values["t_inter_p"]))
        self.ui.irf_sigma.setText(str(default_values["irf_sigma"]))
        self.ui.irf_tau.setText(str(default_values["irf_tau"]))
        self.ui.num_iter.setText(str(default_values["num_iter"]))
        self.ui.img_size.setText(str(default_values["img_size"]))

    def browse_file(self):
        options = QtWidgets.QFileDialog.Option.DontUseNativeDialog
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setOptions(options)
        filters = "MAT Files (*.mat);;JSON Files (*.json)"
        file_name, _ = file_dialog.getOpenFileName(
            self, "Open File", "", filters, options=options
        )
        if file_name:
            self.data_processor.load_data(file_name)

    def run_analysis(self):
        if not self.data_processor.data_loaded:
            QtWidgets.QMessageBox.critical(
                self, "Error", "Please upload data before running the analysis."
            )
            return

        self.ui.run_.setEnabled(False)
        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setValue(0)

        try:
            params = get_values_from_ui(self.ui)
            self.analyzer.run_analysis(
                params, self.data_processor.get_data(), self.update_progress
            )

            QtWidgets.QMessageBox.information(
                self, "Analysis Complete", "The analysis has finished successfully!"
            )
            self.ui.plot_.setEnabled(True)
            self.ui.save_results.setEnabled(True)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"An error occurred during analysis: {str(e)}"
            )
        finally:
            self.ui.run_.setEnabled(True)
            self.ui.progressBar.setVisible(False)

    def update_progress(self, value):
        self.ui.progressBar.setValue(value)
        QtWidgets.QApplication.processEvents()

    def open_hyperparameters(self):
        self.hyperparameters_window = HyperparametersWindow()
        self.hyperparameters_window.parameters_set.connect(self.receive_parameters)
        self.hyperparameters_window.show()

    def receive_parameters(self, params):
        self.analyzer.set_hyperparameters(params)

    def show_plot(self):
        selected_plot = self.ui.combo_plot.currentText()
        plot_data = self.analyzer.get_plot_data()
        self.plot_window = PlotWindow(plot_data, selected_plot)
        self.plot_window.show()

    def export_results(self):
        options = QtWidgets.QFileDialog.Option.DontUseNativeDialog
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setOptions(options)
        filters = "MAT Files (*.mat);;Python Dictionary (*.pkl)"
        file_name, selected_filter = file_dialog.getSaveFileName(
            self, "Save Results", "", filters, options=options
        )
        if file_name:
            format = "mat" if selected_filter == "MAT Files (*.mat)" else "pkl"
            self.data_processor.save_results(
                file_name, self.analyzer.get_results(), format
            )
            QtWidgets.QMessageBox.information(
                self, "Success", f"Results saved successfully as {format} file."
            )
