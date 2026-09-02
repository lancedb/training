"""A Foxglove/Lichtblick layout for a DROID episode: 3 cameras + state/action plots."""

CAMS = [
    ("/observation/images/exterior_1_left", "exterior 1"),
    ("/observation/images/exterior_2_left", "exterior 2"),
    ("/observation/images/wrist_left", "wrist"),
]


def _image(topic):
    return {
        "cameraState": {"distance": 20, "perspective": False,
                        "phi": 60, "thetaOffset": 45,
                        "target": [0, 0, 0], "targetOffset": [0, 0, 0],
                        "targetOrientation": [0, 0, 0, 1], "fovy": 45,
                        "near": 0.5, "far": 5000},
        "followMode": "follow-pose",
        "imageMode": {"imageTopic": topic, "calibrationTopic": None,
                      "synchronize": False, "rotation": 0, "flipHorizontal": False,
                      "flipVertical": False},
        "topics": {}, "layers": {}, "publish": {}, "scene": {},
        "transforms": {}, "cameraTopic": topic,
    }


def _plot(topic, title):
    return {
        "title": title,
        # lerobot's Scalars schema is {scalars: [{label, value}]}; the filtered path
        # `.scalars[:]` expands to one series per feature and Foxglove names each from
        # the message's own `label` field, so we must NOT set a label here.
        "paths": [{"value": f"{topic}.scalars[:].value", "enabled": True,
                   "timestampMethod": "receiveTime"}],
        "showXAxisLabels": True, "showYAxisLabels": True, "showLegend": True,
        "legendDisplay": "floating", "showPlotValuesInLegend": True,
        "isSynced": True, "xAxisVal": "timestamp", "sidebarDimension": 240,
        # seconds of history a LIVE plot shows; 0 is a zero-width window, i.e. blank
        "followingViewWidth": 12,
    }


def build():
    ids = {f"Image!cam{i}": _image(t) for i, (t, _) in enumerate(CAMS)}
    ids["Plot!state"] = _plot("/observation/state", "observation.state")
    ids["Plot!action"] = _plot("/action/state", "action")

    cams_stack = {
        "direction": "column",
        "first": "Image!cam0",
        "second": {"direction": "column", "first": "Image!cam1",
                   "second": "Image!cam2", "splitPercentage": 50},
        "splitPercentage": 34,
    }
    plots_stack = {"direction": "column", "first": "Plot!state",
                   "second": "Plot!action", "splitPercentage": 50}
    return {
        "configById": ids,
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1},
        "layout": {"direction": "row", "first": cams_stack,
                   "second": plots_stack, "splitPercentage": 46},
    }


if __name__ == "__main__":
    import json, sys
    json.dump(build(), sys.stdout, indent=2)
