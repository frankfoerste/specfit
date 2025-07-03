import os
from PyQt6 import QtGui, QtWidgets
from pathlib import Path

def read_lineE(path=None):
    if path is None:
        path = os.path.abspath(__file__)
    path = Path(path)
    parent_dir = str(path.parent)
    lineE = {}
    with open(parent_dir+"/Data/lineE.dat", "r") as lineE_file:
        for line in lineE_file:
            line = line.split()
            lineE[float(line[0])] = [line[1], int(line[2]), line[3], int(line[4]), float(line[5])]
    return lineE

class LineFinder(QtWidgets.QWidget):
    """
    """
    def __init__(self, parent = None, lineE = None, module_path = None):
        self.parent = parent
        super(LineFinder, self).__init__(parent)
        if lineE is None:
            self.lineE = read_lineE(module_path)
        self.screen_properties = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.screen_width = self.screen_properties.width()
        self.screen_height = self.screen_properties.height()
        self.popup_heigth = 220
        self.popup_width = 160
        self.setWindowTitle("Line finder")
        self.setGeometry(self.screen_width-self.popup_width-200, 0, self.popup_width, self.popup_heigth)
        self.table_lines = QtWidgets.QListWidget(self)
        self.init_UI()
        
    def init_UI(self):
        ### initialize widgets ###
        self.table_lines.setGeometry(0, 0, 350, 180) 
        self.show()
        
    def find_nearest_lines(self, energy):
        """
        This function retrieves the nearest item in an array to the given
        value.
        """
        ### clear the QLlistWidget in order to fill it with the possible lines
        ### and to avoid pilling up and pilling up
        self.table_lines.clear()
        energy_low = round(energy,1) - 0.1
        energy_high = round(energy,1) + 0.15
        self.pos_lines = []
        for key, value in self.lineE.items():
            if (key >= energy_low) and (key <= energy_high):
                self.pos_lines.append([value[4],value, key])
        self.pos_lines = sorted(self.pos_lines, reverse = True)
        for i in range(len(self.pos_lines)):
            QtWidgets.QListWidgetItem(
                f"{self.pos_lines[i][2]:.3f} keV - {self.pos_lines[i][1][0]} - {self.pos_lines[i][1][2]}",
                self.table_lines)

    def close_popup(self):
        self.close()
