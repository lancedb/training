"""Layout for the before/after comparison.

One panel per action dimension, each overlaying three series: what the human teleoperator
did, what the 80-step checkpoint predicted, what the 10,000-step checkpoint predicted.
That grouping is the whole point -- the comparison has to be *within* a panel, because
across panels the eye cannot judge which curve tracks the human.
"""

# The publisher sends dims in a fixed order, so scalar index -> action dim is positional.
DIM_ORDER = [0, 5, 6]
DIM_NAMES = {0: "cartesian x", 5: "rotation yaw", 6: "gripper"}

SERIES = [("/action/teleoperator",     "human teleoperator", "#1b1b1b"),
          ("/action/policy_80_steps",  "after 80 steps",     "#d1495b"),
          ("/action/policy_10k_steps", "after 10,000 steps", "#2a9d8f")]


def _image(topic):
    return {"cameraState": {"distance": 20, "perspective": False, "phi": 0, "target": [0,0,0],
            "targetOffset": [0,0,0], "targetOrientation": [0,0,0,1], "thetaOffset": 0, "fovy": 45,
            "near": 0.01, "far": 5000}, "followMode": "follow-pose", "scene": {},
            "transforms": {}, "topics": {}, "layers": {}, "publish": {"type": "point"},
            "imageMode": {"imageTopic": topic}}


def _plot(dim, lo, hi):
    idx = DIM_ORDER.index(dim)
    return {
        "title": f"action dim: {DIM_NAMES[dim]}",
        # One path per series, indexed to THIS dim. `scalars[:]` would splice every dim
        # into a single zigzag series -- legal, plotted, and meaningless.
        "paths": [{"value": f"{topic}.scalars[{idx}].value", "label": label, "color": color,
                   "enabled": True, "timestampMethod": "receiveTime"}
                  for topic, label, color in SERIES],
        "showXAxisLabels": True, "showYAxisLabels": True, "showLegend": True,
        "legendDisplay": "top", "showPlotValuesInLegend": False,
        "isSynced": True, "xAxisVal": "timestamp", "sidebarDimension": 200,
        # seconds of history a LIVE plot shows; 0 is a zero-width window, i.e. blank
        "followingViewWidth": 14,
        # Lichtblick pins y to 0..1 when it cannot infer a range, hiding negative actions.
        "minYValue": lo, "maxYValue": hi,
    }


def build():
    ids = {"Image!cam": _image("/camera"),
           "Plot!d0": _plot(0, -2.8, 4.2),
           "Plot!d6": _plot(6, -2.2, 1.8)}
    plots = {"direction": "column", "first": "Plot!d0", "second": "Plot!d6",
             "splitPercentage": 50}
    return {"configById": ids, "globalVariables": {}, "userNodes": {},
            "playbackConfig": {"speed": 1},
            "layout": {"direction": "row", "first": "Image!cam", "second": plots,
                       "splitPercentage": 36}}
