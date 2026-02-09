"""Isaac Sim bootstrap utilities.

preload_libassimp() must be called BEFORE AppLauncher initialization.
setup_quit_handler() must be called AFTER AppLauncher initialization.
"""

import ctypes
import os
import pathlib


def preload_libassimp() -> None:
    """Preload cmeel-bundled libassimp to prevent symbol clash with Isaac Sim.

    Isaac Sim's asset_converter extension bundles libassimp.so without C++11
    ABI symbols, which conflicts with hpp-fcl (Pinocchio dependency).
    Loading the cmeel-bundled libassimp first prevents the symbol clash.

    Must be called before AppLauncher initialization.
    """
    cmeel_lib = pathlib.Path(
        "/isaac-sim/kit/python/lib/python3.11/site-packages/cmeel.prefix/lib"
    )
    assimp_so = cmeel_lib / "libassimp.so.5"
    if assimp_so.exists():
        ctypes.CDLL(str(assimp_so), mode=ctypes.RTLD_GLOBAL)


def setup_quit_handler() -> list:
    """Register GUI window-close handlers for graceful shutdown.

    Isaac Sim's sim.step() blocks while paused, so a flag-based approach
    cannot work. We terminate the process from inside the callback.

    Must be called after AppLauncher initialization.

    Returns:
        List of event subscriptions (keep alive to prevent GC).
    """
    import omni.kit.app
    from carb.eventdispatcher import get_eventdispatcher

    def _force_quit(_event):
        print("\n[INFO] Window close requested. Shutting down...")
        os._exit(0)

    return [
        get_eventdispatcher().observe_event(
            event_name=omni.kit.app.GLOBAL_EVENT_POST_QUIT,
            on_event=_force_quit,
            observer_name="quit_handler",
            order=0,
        ),
        get_eventdispatcher().observe_event(
            event_name=omni.kit.app.GLOBAL_EVENT_PRE_SHUTDOWN,
            on_event=_force_quit,
            observer_name="shutdown_handler",
            order=0,
        ),
    ]
