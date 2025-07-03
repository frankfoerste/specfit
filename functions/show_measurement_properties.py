import os
from PyQt6 import QtWidgets, QtGui

class ShowProps(QtWidgets.QWidget):
    """
    """
    def __init__(self, data_path, parent = None):
        super(ShowProps, self).__init__(parent)
        self.data_path = data_path
        self.setWindowTitle("Show Measurement Properties")
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        self.setFixedWidth(200)
        self.backup_text = ""
        self.init_UI()
        self.show()
        
    def init_UI(self):
        # initialize widgets
        file_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = "/".join(file_dir.split("/")[:-1])
        self.label_props = QtWidgets.QLabel("Measurement Properties", self)
        self.layout.addWidget(self.label_props, 0, 0, 1, 3)
        self.measurement_properties = QtWidgets.QTextEdit(self)
        self.layout.addWidget(self.measurement_properties, 1, 0, 1, 3)
        self.button_save = QtWidgets.QPushButton("Save", self)
        self.layout.addWidget(self.button_save, 2, 1)
        self.button_save.clicked.connect(self.save_props)
        self.label_logo = QtWidgets.QLabel()
        self.label_logo.setPixmap(QtGui.QPixmap(parent_dir+"/Data/images/specfit_logo_vertical.png"))
        self.label_logo.setScaledContents(True)
        self.label_logo.show() 
        self.layout.addWidget(self.label_logo, 3, 0, 1, 3)
        
    def fill_text(self, string):
        self.measurement_properties.insertPlainText(f"{string}")
        
    def save_props(self):
        with open(f"{self.data_path}/fit_properties.log", "w", encoding = "utf8") as tofile:
            tofile.writelines(str(self.measurement_properties.toPlainText()))
        
    def clear_popup(self):
        self.measurement_properties.clear()
