import pandas as pd
import pretty_midi
import numpy as np
import random
import os


def load_synthetic_dataset(csv_path):
    """
    Load your synthetic dataset.
    """
    return pd.read_csv(csv_path)


def scale_mfcc_to_midi(mfcc_value, mfcc_range=(-400, 400), midi_range=(36, 84)):
    """
    Scale MFCC value to a valid MIDI pitch (36 = C2, 84 = C6).
    """
    return int(np.clip(
        np.interp(mfcc_value, mfcc_range, midi_range),
        midi_range[0],
        midi_range[1]
    ))


def features_to_midi_notes(row, default_tempo=90):
    """
    Convert a row of audio features into a list of MIDI notes.
    """
    tempo = row.get('tempo', default_tempo)
    note_length = 60 / tempo

    pitches = []
    for i in range(1, 14):
        val = row.get(f'mfcc_{i}', None)
        if val is not None:
            pitch = scale_mfcc_to_midi(val)
            pitches.append(pitch)

    # Build note objects
    start_time = 0.0
    notes = []
    for pitch in pitches:
        note = pretty_midi.Note(
            velocity=random.randint(70, 100),
            pitch=pitch,
            start=start_time,
            end=start_time + note_length
        )
        notes.append(note)
        start_time += note_length

    return notes


def save_notes_to_midi(notes, output_file, program=0):
    """
    Save a list of notes to a MIDI file.
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    instrument.notes.extend(notes)
    midi.instruments.append(instrument)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    midi.write(output_file)
    print(f"✅ MIDI file saved: {output_file}")


def generate_lofi_midi_from_synthetic_row(csv_path, output_file='generated/lofi_beat.mid', row_index=None):

    df = load_synthetic_dataset(csv_path)

    # Pick a row (random or specific)
    if row_index is None:
        row = df.sample(1).iloc[0]
    else:
        row = df.iloc[row_index]

    notes = features_to_midi_notes(row)
    save_notes_to_midi(notes, output_file)

    return output_file
