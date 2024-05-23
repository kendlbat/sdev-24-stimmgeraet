from typing import List, Tuple
from enum import Enum


# Type that maps notes to their frequencies
class NoteFreq(Enum):
    C0 = 16.35
    C0S = 17.32
    D0 = 18.35
    D0S = 19.45
    E0 = 20.60
    F0 = 21.83
    F0S = 23.12
    G0 = 24.50
    G0S = 25.96
    A0 = 27.50
    A0S = 29.14
    B0 = 30.87
    C1 = 32.70
    C1S = 34.65
    D1 = 36.71
    D1S = 38.89
    E1 = 41.20
    F1 = 43.65
    F1S = 46.25
    G1 = 49.00
    G1S = 51.91
    A1 = 55.00
    A1S = 58.27
    B1 = 61.74
    C2 = 65.41
    C2S = 69.30
    D2 = 73.42
    D2S = 77.78
    E2 = 82.41
    F2 = 87.31
    F2S = 92.50
    G2 = 98.00
    G2S = 103.83
    A2 = 110.00
    A2S = 116.54
    B2 = 123.47
    C3 = 130.81
    C3S = 138.59
    D3 = 146.83
    D3S = 155.56
    E3 = 164.81
    F3 = 174.61
    F3S = 185.00
    G3 = 196.00
    G3S = 207.65
    A3 = 220.00
    A3S = 233.08
    B3 = 246.94
    C4 = 261.63
    C4S = 277.18
    D4 = 293.66
    D4S = 311.13
    E4 = 329.63
    F4 = 349.23
    F4S = 369.99
    G4 = 392.00
    G4S = 415.30
    A4 = 440.00
    A4S = 466.16
    B4 = 493.88
    C5 = 523.25
    C5S = 554.37
    D5 = 587.33
    D5S = 622.25
    E5 = 659.26
    F5 = 698.46
    F5S = 739.99
    G5 = 783.99
    G5S = 830.61
    A5 = 880.00
    A5S = 932.33
    B5 = 987.77
    C6 = 1046.50
    C6S = 1108.73
    D6 = 1174.66
    D6S = 1244.51
    E6 = 1318.51
    F6 = 1396.91
    F6S = 1479.98
    G6 = 1567.98
    G6S = 1661.22
    A6 = 1760.00
    A6S = 1864.66
    B6 = 1975.53
    C7 = 2093.00
    C7S = 2217.46
    D7 = 2349.32
    D7S = 2489.02
    E7 = 2637.02
    F7 = 2793.83
    F7S = 2959.96
    G7 = 3135.96
    G7S = 3322.44
    A7 = 3520.00
    A7S = 3729.31
    B7 = 3951.07
    C8 = 4186.01
    C8S = 4434.92
    D8 = 4698.63
    D8S = 4978.03
    E8 = 5274.04
    F8 = 5587.65
    F8S = 5919.91
    G8 = 6271.93
    G8S = 6644.88
    A8 = 7040.00
    A8S = 7458.62
    B8 = 7902.13


def hertz_to_note(hertz: float) -> Tuple[NoteFreq, float]:
    lowest_diff = float("inf")
    closest_note = None
    offset = 0
    for note in NoteFreq:
        diff = abs(note.value - hertz)
        if diff < lowest_diff:
            lowest_diff = diff
            closest_note = note
            offset = note.value - hertz
    return closest_note, offset


def main():
    test = [
        440.0,
        235.0,
        500.0,
        890.0,
    ]

    for hertz in test:
        note, offset = hertz_to_note(hertz)
        print(f"Frequency: {hertz} Hz")
        print(f"Closest note: {note.name}")
        print(f"Offset: {offset:.2f} Hz")


if __name__ == "__main__":
    main()
