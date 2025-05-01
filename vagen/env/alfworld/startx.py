#!/usr/bin/env python3
"""
start_x11.py

A self-contained script to launch a virtual X11 server using NVIDIA and Xorg,
enabling AI2-THOR to run headlessly on your remote server.

Usage:
    python start_x11.py [DISPLAY]
    # e.g. python start_x11.py 0  <-- will start X on :0 and export DISPLAY

You can also import and call start_x11.start(display) from Python before
initializing your ALFWorldEnv or AI2-THOR Controller.
"""
import os
import re
import shlex
import platform
import subprocess
import tempfile

# =============================================================================
# Utilities to suppress ALSA errors by writing a null asound configuration
# =============================================================================
def _setup_null_asoundrc():
    """
    Create a ~/.asoundrc file that directs ALSA to the null device,
    preventing 'cannot find card' errors.
    """
    home = os.path.expanduser('~')
    cfg_path = os.path.join(home, '.asoundrc')
    if os.path.exists(cfg_path):
        return
    null_cfg = (
        "pcm.!default {\n"
        "    type null\n"
        "}\n"
        "ctl.!default {\n"
        "    type null\n"
        "}\n"
    )
    try:
        with open(cfg_path, 'w') as f:
            f.write(null_cfg)
    except Exception:
        # If writing fails (e.g., permission), silently ignore
        pass

# =============================================================================
# Core X11 startup logic
# =============================================================================
def pci_records():
    """Parse `lspci -vmm` output into list of dicts."""
    out = subprocess.check_output(shlex.split('lspci -vmm')).decode()
    recs = []
    for device in out.strip().split("\n\n"):
        d = {}
        for row in device.split("\n"):
            key, val = row.split("\t")
            d[key.split(':')[0]] = val
        recs.append(d)
    return recs


def generate_xorg_conf(bus_ids, width=1280, height=1024):
    """Generate an Xorg config that uses NVIDIA GPUs headlessly."""
    device_tpl = '''Section "Device"
    Identifier  "Device{idx}"
    Driver      "nvidia"
    VendorName  "NVIDIA Corporation"
    BusID       "{bus}"
EndSection
'''
    screen_tpl = '''Section "Screen"
    Identifier  "Screen{idx}"
    Device      "Device{idx}"
    DefaultDepth 24
    Option      "AllowEmptyInitialConfiguration" "True"
    SubSection "Display"
        Depth 24
        Virtual {w} {h}
    EndSubSection
EndSection
'''
    parts = []
    layout_lines = []
    for i, bus in enumerate(bus_ids):
        parts.append(device_tpl.format(idx=i, bus=bus))
        parts.append(screen_tpl.format(idx=i, w=width, h=height))
        layout_lines.append(f"    Screen {i} \"Screen{i}\" 0 0")
    layout = (
        "Section \"ServerLayout\"\n"
        "    Identifier \"Layout0\"\n" +
        "\n".join(layout_lines) +
        "\nEndSection\n"
    )
    return "\n".join(parts) + "\n" + layout


def start(display=0, width=1280, height=1024):
    """Launch a headless X server on the given DISPLAY index."""
    if platform.system() != 'Linux':
        raise RuntimeError("start_x11 only supports Linux")

    # suppress ALSA errors
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    _setup_null_asoundrc()

    # find NVIDIA GPUs
    buses = []
    for r in pci_records():
        if r.get('Vendor') == 'NVIDIA Corporation' and r.get('Class','').startswith('VGA'):
            slot = r['Slot']  # e.g. '01:00.0'
            parts = re.split(r'[:\.]', slot)
            buses.append('PCI:' + ':'.join(str(int(x,16)) for x in parts))

    if not buses:
        raise RuntimeError("No NVIDIA GPU found for Xorg virtual display")

    # write temporary xorg.conf
    fd, path = tempfile.mkstemp(suffix='.conf')
    conf = generate_xorg_conf(buses, width, height)
    with os.fdopen(fd, 'w') as f:
        f.write(conf)

    # launch Xorg in the foreground
    cmd = (
        f"Xorg -noreset +extension GLX +extension RANDR +extension RENDER "
        f"-config {path} :{display}"
    )
    process = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print(f"Started Xorg on DISPLAY=:{display}")

    # wait for Xorg process to complete (or manually stop it)
    out, err = process.communicate()

    if process.returncode != 0:
        print(f"Error starting Xorg: {err.decode()}")
        return

    # export DISPLAY for this process
    os.environ['DISPLAY'] = f":{display}"


if __name__ == '__main__':
    import sys
    disp = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start(disp)
    print(f"Use DISPLAY=:{disp} for your AI2-THOR processes.")