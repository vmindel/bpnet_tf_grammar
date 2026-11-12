"""Common figure settings: editable text in PDF/SVG (Illustrator), output folder."""
import matplotlib.pyplot as plt
import paths

plt.rcParams['pdf.fonttype'] = 42    # TrueType text, editable in Illustrator
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
paths.FIGURES.mkdir(exist_ok=True)


def fig_path(name):
    return str(paths.FIGURES / name)
