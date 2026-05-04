"""
Diagnostic: dump JSF message 80 payload as multiple integer interpretations.

Reads the first ping of each (subsystem, channel) combination from a single
JSF file, then for each byte offset prints the value interpreted as:
  - int16 / uint16 (little-endian)
  - int32 / uint32 (little-endian)
  - float32 (little-endian)

Use this to find offsets where the operator's pulse length, range setting,
transmit level, etc. live. The EdgeTech spec varies between firmware
revisions, so trust the dump rather than any single spec PDF.

Usage: just run; edit JSF_PATH below if needed.
"""
import struct
from pathlib import Path

from src.config import ROOT


# pick any one jsf file; first ping of each subsystem will be dumped
JSF_PATH = next((ROOT / "data/sss/20251223").glob("*.jsf"))

# dump first ping per subsystem (LF=20, HF=21) per side (port=0, stbd=1)
TARGETS = [(20, 0), (20, 1), (21, 0), (21, 1)]

# offsets we already know about (from src/sss/read_sss_jsf.py),
# annotated so they stand out in the dump
KNOWN_OFFSETS = {
    0:   "ping_time_sec (uint32)",
    30:  "validity flag (uint16)",
    48:  "heave (float32)",
    80:  "rx coord X (int32)",
    84:  "rx coord Y (int32)",
    88:  "coord units (int16)",
    116: "sample_interval_ns (uint32, => pix_m)",
    126: "start_freq_x10 (uint16)",
    128: "end_freq_x10 (uint16)",
    136: "depth (int32, /1000 = m)",
    152: "center_freq_hz (float32)",
    168: "weight_factor (int16, 2^-N scale)",
    200: "ping_time_micro (uint32)",
}

# ranges that are "physically plausible" for typical SSS settings -
# values landing in these ranges across multiple offsets are candidates
# worth investigating further
PLAUSIBLE_RANGES = {
    "pulse_length_samples": (50, 4000),
    "range_setting_decimeters": (200, 2000),  # 20-200 m
    "transmit_level_pct": (0, 100),
    "starting_depth_samples": (0, 5000),
    "ADC_gain": (0, 100),
}


def dump_payload(pay, subsys, channel):
    print(f"\n{'='*72}")
    print(f"Subsystem={subsys} ({'LF' if subsys == 20 else 'HF'}), "
          f"Channel={channel} ({'port' if channel == 0 else 'stbd'})")
    print(f"Payload size: {len(pay)} bytes")
    print(f"{'='*72}")
    print(f"{'offset':>6} {'u16':>8} {'i16':>8} {'u32':>12} {'i32':>12} "
          f"{'f32':>14}  notes")
    print("-" * 90)

    for off in range(0, min(240, len(pay) - 4), 2):
        try:
            u16 = struct.unpack_from("<H", pay, off)[0]
            i16 = struct.unpack_from("<h", pay, off)[0]
            u32 = struct.unpack_from("<I", pay, off)[0] if off + 4 <= len(pay) else None
            i32 = struct.unpack_from("<i", pay, off)[0] if off + 4 <= len(pay) else None
            f32 = struct.unpack_from("<f", pay, off)[0] if off + 4 <= len(pay) else None
        except struct.error:
            continue

        # Note column
        note = KNOWN_OFFSETS.get(off, "")

        # Flag plausible candidates only when offset is unknown
        if not note:
            for name, (lo, hi) in PLAUSIBLE_RANGES.items():
                if lo <= u16 <= hi or (i16 > 0 and lo <= i16 <= hi):
                    note = f"  ← maybe {name}? (u16={u16})"
                    break

        # Filter: skip rows where everything is 0 unless it's a known offset
        if not note and u16 == 0 and i16 == 0:
            continue

        f32_str = f"{f32:>14.4g}" if f32 is not None and abs(f32) < 1e20 else " " * 14
        u32_str = f"{u32:>12}" if u32 is not None else " " * 12
        i32_str = f"{i32:>12}" if i32 is not None else " " * 12
        print(f"{off:>6} {u16:>8} {i16:>8} {u32_str} {i32_str} {f32_str}  {note}")


def main():
    print(f"Source file: {JSF_PATH}")

    found = {tgt: False for tgt in TARGETS}
    file_size = JSF_PATH.stat().st_size

    with open(JSF_PATH, "rb") as f:
        while f.tell() < file_size and not all(found.values()):
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            if struct.unpack_from("<H", hdr, 0)[0] != 0x1601:
                f.seek(-15, 1)
                continue

            msg_type = struct.unpack_from("<H", hdr, 4)[0]
            subsys = hdr[7]
            channel = hdr[8]
            pay_size = struct.unpack_from("<I", hdr, 12)[0]
            pay = f.read(pay_size)

            tgt = (subsys, channel)
            if msg_type == 80 and tgt in TARGETS and not found[tgt]:
                dump_payload(pay, subsys, channel)
                found[tgt] = True

    missing = [tgt for tgt, ok in found.items() if not ok]
    if missing:
        print(f"\nMissing targets (no msg=80 found): {missing}")


if __name__ == "__main__":
    main()