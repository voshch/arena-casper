from __future__ import annotations

import importlib.util

import pytest

_ROS_SKIP_REASON = "ROS2 not discoverable: source install/setup.bash to enable"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if importlib.util.find_spec("rclpy") is not None:
        return
    skip = pytest.mark.skip(reason=_ROS_SKIP_REASON)
    for item in items:
        if "/tests/ros/" in str(item.path):
            item.add_marker(skip)
