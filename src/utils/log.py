from PyQt6 import QtGui

class Logger:
    def __init__(self, edit, out=None, color=QtGui.QColor(0,0,0)):
        self.edit = edit
        self.out = None
        self.color = color

    def write(self, m):
        m = "-> " + m

        self.edit.setTextColor(self.color)
        self.edit.append(m)
        if self.out:
            self.out.write(m)
