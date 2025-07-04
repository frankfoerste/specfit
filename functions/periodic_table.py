import sys
import os
import numpy as np
from functools import partial
from PyQt6 import QtGui, QtWidgets, QtCore
import xraylib as xrl
sys.path.append("functions")
from user_defined_lines import LinesWidget
import line_finder as lf

class PeriodicTable(QtWidgets.QWidget):
    def __init__(self,
                 parent = None,
                 PSE = None,
                 elements = None,
                 lines = None,
                 check_elements = None,
                 characteristic_lines = None,
                 lineE = None):
        # initalize main window
        super(PeriodicTable, self).__init__(parent)
        try:
            QtGui.QIcon.setThemeName("ubuntu-mono-dark")
        except:
            print(QtGui.QIcon.themeName())
        self.bg_color = "light grey"
        # implementing database
        self.elements = elements  # list of elements
        self.lines = lines  # list of lines for ex. K, Ka,..
        self.PSE = PSE  # list with definitions of the PSE
        self.characteristic_lines = characteristic_lines  # list of characteristic X-ray lines
        self.check_lines_list = []  # list of checkable lines
        self.check_line_labels = []  # list of check_line_labels
        self.check_elements = check_elements  # list to determine the selected elements
        self.lineE = lineE
        # define geometry and properties
        self.screen_properties = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.screen_width = self.screen_properties.width()
        self.screen_height = self.screen_properties.height()
        self.window_heigth = 320
        self.window_width = 365
        self.setWindowTitle("Periodic Table")
        self.setGeometry((self.screen_width-self.window_width)//2,
                         (self.screen_height-self.window_heigth)//2,
                         self.window_width, self.window_heigth)
        self.selected_elements = []
        self.selected_lines = []
        self.user_defined_lines = []
        self.setStyleSheet("QWidget { color: black; background-color:white;}"\
                           +"QLabel {font-size: 11px;} "\
                           +"QLineEdit {font-size: 11px} "\
                           +"QCheckBox {font-size: 10px} "\
                           +"QPushButton {font-size: 11px;}"\
                           +"QRadioButton {font-size: 11px} "\
                           +"QTextEdit {font-size: 11px}"\
                           +"QTabWidget {font-size: 11px}" )
        self.__init__UI()

    def __init__UI(self):
        """
        define the layout of the UI
        """
        # create QVBoxLayout
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)
        # create tab widget
        self.tabs = QtWidgets.QTabWidget()
        self.__init__PSE_widget()
        self.__init__line_finder_widget()
        self.__init__udl_widget()
        self.tabs.insertTab(0, self.tab_PSE, "PSE")
        self.tabs.insertTab(1, self.tab_line_finder, "line finder")
        self.tabs.insertTab(2, self.tab_udl, "user defined lines")
        self.__init__selected_elements_widget()
        # add widgets to QVBoxLayout
        self.layout.addWidget(self.tabs, 0, 0)
        self.layout.addWidget(self.scroll_elements, 1, 0)

    def __init__selected_elements_widget(self):
        self.element_widget = QtWidgets.QWidget()
        self.scroll_elements = QtWidgets.QScrollArea()
        self.scroll_elements.setWidget(self.element_widget)

    def __init__line_finder_widget(self):
        module_path = os.path.dirname(lf.__file__)
        self.tab_line_finder = lf.LineFinder(parent = None, module_path = module_path)
        self.tab_line_finder.table_lines.doubleClicked.connect(self.select_line_on_double_click_line_finder)

    def __init__PSE_widget(self):
        self.tab_PSE = QtWidgets.QWidget()
        self.PSE_layout = QtWidgets.QGridLayout()
        self.PSE_layout.setSpacing(0)
        self.tab_PSE.setLayout(self.PSE_layout)
        self.element_list()
        self.label_lanthanides = QtWidgets.QLabel("Lanthanides", self.tab_PSE)
        self.label_actinides= QtWidgets.QLabel("Actinides", self.tab_PSE)
        self.PSE_layout.addWidget(self.label_lanthanides, 7, 0, 1, 3)
        self.PSE_layout.addWidget(self.label_actinides, 8, 0, 1, 3)
        self.label_lanthanides.setStyleSheet("QWidget {font-size: 9px;}")
        self.label_actinides.setStyleSheet("QWidget {font-size: 9px;}")
        self.tabs.setFixedSize(365, 210)

    def __init__udl_widget(self):
        self.tab_udl = LinesWidget()

    def show_periodic_table(self):
        self.show()

    def update_element_widget(self):
        self.element_widget.resize(365, 20*len(self.check_lines_list))
        self.scroll_elements.setWidget(self.element_widget)

    def clear_element_line_list(self):
        self.clear_line_list()
        self.selected_elements = []
        self.selected_lines = []

    def element_list(self):
        """
        This function creates checkbuttons for each element. You can select the
        elements, which then will be saved in the list selected_elements in the form
        [[element_1,Z_1],[element_2,Z_2],...]
        """
        self.selected_elements = []
        for i, _ in enumerate(self.elements):
            self.check_elements[i] = QtWidgets.QPushButton(self.elements[i][1],self.tab_PSE)
            self.check_elements[i].setCheckable(True)
            self.PSE_layout.addWidget(self.check_elements[i], self.PSE[i][3],self.PSE[i][2])
            self.check_elements[i].setStyleSheet("QWidget {font-size: 11px; background-color : %s}"%self.PSE[i][5])
            self.check_elements[i].pressed.connect(partial(self.read_element_choice,self.elements[i][1]))
            self.check_elements[i].setToolTip(self.create_tool_tip_element(i+1))

    def create_tool_tip_element(self, element):
        element = str(element)
        tool_tip = ""
        for k, _ in enumerate(self.characteristic_lines[element]):
            tool_tip += f"{self.characteristic_lines['No.'][k]} \t {self.characteristic_lines[element][k]} \n"
        return tool_tip

    def display_selected_lines(self):
       """
       display lines that are stored in dataclass in GUI -necessary after loading a paramfile
       """
       # add element-lines
       for element,lines in zip(self.selected_elements, self.selected_lines):
             self.read_element_choice(element,paramfile= True)
             self.select_lines_in_list(lines)
       # add user-defined-lines
       for element_str,start,end in self.user_defined_lines:
             self.tab_udl.new_filled_line(element_str, start, end)

    def get_selected_lines(self):
        """
        This function reads out the lines checkbuttons and saves it into the
        selected_lines list.
        """
        linedict = {1: "K", 2: "Ka", 3: "Kb",
                    4: "L", 5: "L1", 6: "L2", 7: "L3",
                    8: "M", 9: "M1", 10: "M2", 11: "M3", 12: "M4", 13: "M5"}
        self.selected_elements = []
        self.selected_lines = []
        for lines in self.check_lines_list:
            self.selected_elements.append(lines[0].text()) ##TODO
            line_list = []  # store for example ['Ka','Kb']
            for j in range(1,14):
                if lines[j].checkState() == QtCore.Qt.CheckState.Checked:
                    line_list.append(linedict[j])
            self.selected_lines.append(line_list)

    def read_element_choice(self, element_str, paramfile=False):
        """
        This function determines wether the element-button was already pressed
        or the element must be assigned to the selected_element list (appear in the GUI).
        """
        if paramfile or not (element_str in self.selected_elements):
            if not paramfile:
                self.selected_elements.append(element_str)
            self.line_list(element_str)
            element = xrl.SymbolToAtomicNumber(element_str)
            if element < 13:
                Ka = [xrl.LineEnergy(element, xrl.KA1_LINE), xrl.RadRate(element, xrl.KA1_LINE)]
                self.ax_canvas_spectrum.axvline(Ka[0],
                                                linewidth=0.5, color="r",
                                                ymax=Ka[1], label=element_str)
                self.ax_canvas_spectrum.text(Ka[0] + 0.1,
                                             0.3, f"{element_str} - K", rotation=90, color="green",
                                             label=element_str)
            elif 13 <= element <= 39:
                Ka = [xrl.LineEnergy(element, xrl.KA1_LINE), xrl.RadRate(element, xrl.KA1_LINE)]
                Kb = [xrl.LineEnergy(element, xrl.KB1_LINE), xrl.RadRate(element, xrl.KB1_LINE)]
                self.ax_canvas_spectrum.axvline(Ka[0],
                                                linewidth=0.5, color="r", ymax=Ka[1],
                                                label=element_str)
                self.ax_canvas_spectrum.text(Ka[0] + 0.1,
                                             0.3, f"{element_str} - K", rotation=90, color="green",
                                             label=element_str)
                self.ax_canvas_spectrum.axvline(Kb[0],
                                                linewidth=0.5, color="r", ymax=Kb[1],
                                                label=element_str)
            elif 39 < element < 96:
                self.ax_canvas_spectrum.axvline(xrl.LineEnergy(element, xrl.L3M5_LINE),
                                                linewidth=0.5, color="r", ymax=0.5,
                                                label=element_str)
                self.ax_canvas_spectrum.text(xrl.LineEnergy(element, xrl.L3M5_LINE) + 0.1,
                                             0.3, f"{element_str} - La", rotation=90,
                                             label=element_str)
            else:
                self.ax_canvas_spectrum.axvline(xrl.LineEnergy(element, xrl.M5N7_LINE),
                                                linewidth=0.5, color="r", ymax=0.5,
                                                label=element_str)
                self.ax_canvas_spectrum.text(xrl.LineEnergy(element, xrl.M5N7_LINE) + 0.1,
                                             0.3, f"{element_str} - Ma", rotation=90,
                                             label=element_str)
            self.canvas_spectrum.draw()
        else:
            self.remove_line_list(element_str)
            i = np.where(np.array(self.selected_elements) == element_str)[0][0]
            self.selected_elements.pop(i)
        self.update_element_widget()

    def line_list(self, element_str):
        """
        This function creates checkbuttons for each line. You can select the
        lines, which then will be saved in the list selected_lines in the form
        [line_1,line_2,...]
        """
        if self.check_lines_list == []:
            self.check_lines_list = [[element_str, xrl.SymbolToAtomicNumber(element_str), element_str,
                                      0, 1, 2,
                                      3, 4, 5, 6,
                                      7, 8, 9, 10, 11, 12]]
        else:
            self.check_lines_list.append([element_str, xrl.SymbolToAtomicNumber(element_str), element_str,
                                          0, 1, 2,
                                          3, 4, 5, 6,
                                          7, 8, 9, 10, 11, 12])
        len_list = len(self.check_lines_list)
        self.check_lines_list[-1][0] = QtWidgets.QPushButton(element_str, self.element_widget)
        self.check_lines_list[-1][0].setGeometry(0, 0 + 20*(len_list-1), 20, 20)
        self.check_lines_list[-1][0].clicked.connect(partial(self.read_element_choice, element_str))
        self.check_lines_list[-1][0].show()
        for i in range(1, 14):
            self.check_lines_list[-1][i] = QtWidgets.QCheckBox(f"{self.lines[i - 1][0]}", self.element_widget)
            self.check_lines_list[-1][i].clicked.connect(partial(self.Lines_separate, i))
            if i == 1 and xrl.SymbolToAtomicNumber(element_str) <= 39:
                self.check_lines_list[-1][i].setChecked(True)
            elif 4 == i and 39 < xrl.SymbolToAtomicNumber(element_str):
                self.check_lines_list[-1][i].setChecked(True)
            self.check_lines_list[-1][i].setGeometry(25+ 36*(i-1), 0 + 20*(len_list-1), 33, 20)
            self.check_lines_list[-1][i].show()
        self.update_element_widget()

    def Lines_separate(self, lineindex): #1:K 2:Ka 3:Kb
        """
        prohibits that
        K and Ka, Kb
        L and L1, L2, L3
        M and M1, M2, M3, M4, M5
        can be checked together
        """
        for i, _ in enumerate(self.check_lines_list):
            # K
            if lineindex == 1:
                if self.check_lines_list[i][1].checkState().value != 0:
                    self.check_lines_list[i][2].setChecked(False)
                    self.check_lines_list[i][3].setChecked(False)
            elif (lineindex == 2 or lineindex == 3):
                if self.check_lines_list[i][lineindex].checkState().value != 0:
                    self.check_lines_list[i][1].setChecked(False)
            # L
            elif lineindex == 4:
                if self.check_lines_list[i][4].checkState().value != 0:
                    self.check_lines_list[i][5].setChecked(False)
                    self.check_lines_list[i][6].setChecked(False)
                    self.check_lines_list[i][7].setChecked(False)
            elif (lineindex == 5 or lineindex == 6 or lineindex == 7):
                if self.check_lines_list[i][lineindex].checkState().value != 0:
                    self.check_lines_list[i][4].setChecked(False)
            # M
            elif lineindex == 8:
                if self.check_lines_list[i][8].checkState().value != 0:
                    self.check_lines_list[i][9].setChecked(False)
                    self.check_lines_list[i][10].setChecked(False)
                    self.check_lines_list[i][11].setChecked(False)
                    self.check_lines_list[i][12].setChecked(False)
                    self.check_lines_list[i][13].setChecked(False)
            elif (lineindex == 9 or lineindex == 10 or lineindex == 11 or lineindex == 12 or lineindex == 13):
                if self.check_lines_list[i][lineindex].checkState().value != 0:
                    self.check_lines_list[i][8].setChecked(False)

    def select_lines_in_list(self,linelist):
       """
       selects the lines in linelist in a previousy created line_list of latest added element
       """
       linedict = {"K": 1, "Ka": 2, "Kb": 3,
                   "L": 4, "L1": 5, "L2": 6, "L3": 7,
                   "M": 8, "M1": 9, "M2": 10, "M3":11,"M4": 12, "M5": 13}
       for line in linedict.keys():
           self.check_lines_list[-1][linedict[line]].setChecked(line in linelist)

    def remove_line_list(self, element_str):
        """
        remove the line with element label from GUI
        """
        i = np.where(np.array(self.selected_elements) == element_str)[0][0]
        for j in range(14):
            self.check_lines_list[i][j].hide()  # hide all the buttons for element_nr
        self.check_lines_list.pop(i)  # remove this element of check_lines

        for k, _ in enumerate(self.check_lines_list):  # find the correct position for all other
            for l in range(14):
                if l == 0:
                    self.check_lines_list[k][0].move(0, 0 + 20*k)
                if (l >= 1 and l <= 13):
                    self.check_lines_list[k][l].move(25+ 36*(l-1), 0+ 20*(k))
                self.check_lines_list[k][l].show()
        # now remove the vertical lines from the specfit plot canvas
        tidy_lines = []
        tidy_texts = []
        for line in self.ax_canvas_spectrum.lines:
            if line.properties()["label"] != element_str:
                tidy_lines.append(line)
        for text in self.ax_canvas_spectrum.texts:
            if text.properties()["label"] != element_str:
                tidy_texts.append(text)
        try:
            self.ax_canvas_spectrum.lines = tidy_lines
            self.ax_canvas_spectrum.texts = tidy_texts
        except:
            print("Could not set lines in canvas")
        self.canvas_spectrum.draw()
        self.update_element_widget()

    def clear_line_list(self):
        """
        This function removes all lines-checkbuttons
        """
        for i, _ in enumerate(self.check_lines_list):
            for j in range(14):
                self.check_lines_list[i][j].hide()
        self.check_lines_list = []
        for i, _ in enumerate(self.elements):
            self.check_elements[i].setChecked(False)
        self.update_element_widget()

    def fill_and_show_lines_widget(self):
        """
        load current spec and params in lines widget and display it
        """
        self.tab_udl.display_lines_widget()

    def build_specfit_addlines(self):
        #builds dict{'element':(True/False, ['linename',...], Z), '':...,}
        # example: {'Cu': (True, ['K-line', 'L1'], 29)}
        self.specfit_addlines = {}
        for element, lines in zip(self.selected_elements, self.selected_lines):
            self.specfit_addlines[f"{element}"] = (True, lines, xrl.SymbolToAtomicNumber(element))

    def select_line_on_double_click_line_finder(self, ):
        selected_line = self.tab_line_finder.table_lines.currentItem().text()
        element = selected_line.split()[3]
        line = selected_line.split()[5]
        self.check_elements[xrl.SymbolToAtomicNumber(element)-1].click()
        for i, _ in enumerate(self.check_lines_list):
            if element == self.check_lines_list[i][0].text():
                for j in range(len(self.check_lines_list[i])-3):
                    if self.check_lines_list[i][j+1].text() == line:
                        self.check_lines_list[i][j+1].setChecked(True)
                    else:
                        self.check_lines_list[i][j+1].setChecked(False)

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("GTK+")
    pertab = PeriodicTable()
    pertab.show()
    app.exec_()

if __name__ == "__main__":
    main()
