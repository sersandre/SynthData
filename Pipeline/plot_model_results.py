import matplotlib.pyplot as plt
import numpy as np
import json

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                spine.set_visible(False)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def example_data():
    synthetic_classes = ["eiba_0_10", "eiba_0_19", "eiba_0_5", "eiba_1_1",
                         "eiba_1_13", "eiba_1_9", "eiba_2_7", "eiba_4_11",
                         "eiba_4_27", "eiba_4_6", "eiba_4_9", "eiba_0_14", 
                         "eiba_0_2", "eiba_0_6", "eiba_1_10",
                         "eiba_2_10", "eiba_2_8", "eiba_4_24", "eiba_4_29",
                         "eiba_4_7", "eiba_5_2"]
    dataset = "results_MVIP.json"

    data = [synthetic_classes]
    dataset = f"results_MVIP.json"
    with open(dataset, "r") as f:
        raw_data = json.load(f)
    acc = []
    acc_3 = []
    for key in synthetic_classes:
        acc.append(float(raw_data[key]["top1_ACC"]) / 100.0)
        acc_3.append(float(raw_data[key]["top3_ACC"]) / 100.0)
    dataset = f"results_MVIP_synthetic_real_mixed.json"
    with open(dataset, "r") as f:
        raw_data = json.load(f)
    acc1 = []
    acc1_3 = []
    for key in synthetic_classes:
        acc1.append(float(raw_data[key]['top1_ACC']) / 100.0)
        acc1_3.append(float(raw_data[key]['top3_ACC']) / 100.0)
    data.append(("", [acc, acc_3, acc1, acc1_3]))
    print(data)
    return data


if __name__ == '__main__':
    data = example_data()

    N = len(data[0])
    theta = radar_factory(N, frame='polygon')


    spoke_labels = data.pop(0)
    title, case_data = data[0]

    _min, _max = min(case_data[0]), max(case_data[-1])

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    ax.set_rgrids([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_title(title, position=(0.5, 1.1), ha='center')
    #ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)

    colors = ["#E00922", "#EDB32B", "#424EF5", "#0BE0CF"]
    labels = ("Baseline Top1_ACC", "Baseline Top3_ACC", "Gemischt Top1_ACC", "Gemitsch Top3_ACC")


    for d, color in zip(case_data, colors):
        line = ax.plot(theta, d, color=color)
        ax.fill(theta, d, alpha=0.25, facecolor=color, label='_nolegend_')
    ax.set_varlabels(spoke_labels)

    #for t, r in zip(theta, case_data[-1]):
    #    ax.annotate("{}".format(r), xy=[t, r], fontsize=10)

    legend = ax.legend(labels, loc=(0.9, 0.95), labelspacing=0.1, fontsize='small')
    plt.title("Vergleich Baseline/Gemischt", fontsize='large', weight='bold')

    plt.show()



    plt.show()