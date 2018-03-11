import os

# automatic compilation of .ui and .rcc files
os.execvp("pyuic5", ["pyuic5", "../ui/airs_window.ui", ">", "../ui/airs_window.py"])
