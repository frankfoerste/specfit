from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon


# functions to maintain and handle themes for the SpecFit GUI

def load_theme(filename):
    """
    Load the theme from the specfit/themes folder

    Parameters
    ----------
    filename : pathlib.Path
        Path to the theme file to load

    """
    with open(filename, "r", encoding="utf-8") as file:
        QApplication.instance().setStyleSheet(file.read())


class ThemeManager(QObject):
    theme_changed = pyqtSignal(str)

    def __init__(
            self,
            app: QApplication,
            theme_dir: str = "themes",
            ):
        super().__init__()

        self.app = app
        self.theme_dir = Path(theme_dir)

        self.current_theme = "light"

    def set_theme(
            self,
            theme: str
            ):
        
        if theme not in ("light", "dark"):
            raise ValueError(f"Unknown theme: {theme}")

        if theme == self.current_theme:
            return

        qss_file = self.theme_dir / f"{theme}.qss"

        if not qss_file.exists():
            raise FileNotFoundError(qss_file)

        stylesheet = qss_file.read_text(encoding="utf-8")

        self.app.setStyleSheet(stylesheet)
        
        self.current_theme = theme

        self.theme_changed.emit(theme)

    def toggle(self):
        if self.current_theme == "light":
            self.set_theme("dark")
        else:
            self.set_theme("light")

    def is_dark(self):
        return self.current_theme == "dark"
