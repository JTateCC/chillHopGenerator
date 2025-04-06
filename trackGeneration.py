from midi2audio import FluidSynth
from pydub import AudioSegment
import pretty_midi
import numpy as np
import random
import os
def convert_midi_to_wav(midi_path, wav_path, soundfont_path='soundfonts/lofi.sf2'):
    fs = FluidSynth(soundfont_path)
    fs.midi_to_audio(midi_path, wav_path)
    return wav_path


def add_lofi_effects(wav_path, output_path, crackle_path='samples/vinyl_crackle.mp3'):
    base_track = AudioSegment.from_file(wav_path)
    crackle = AudioSegment.from_file(crackle_path).low_pass_filter(1000)

    # Loop crackle if it's shorter than the track
    while len(crackle) < len(base_track):
        crackle += crackle

    crackle = crackle[:len(base_track)] - 10  # Reduce volume
    final_mix = base_track.overlay(crackle)

    # Add lo-fi warmth
    final_mix = final_mix.low_pass_filter(3000).fade_in(500).fade_out(500)
    final_mix.export(output_path, format="mp3")
    return output_path

def synthetic_row_to_midi(row, output_file='generated/lofi.mid', tempo_override=None):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    tempo = row.get('tempo', tempo_override or 85)
    note_length = 60 / tempo

    pitches = []
    for i in range(1, 14):
        val = row.get(f'mfcc_{i}', None)
        if val is not None:
            pitch = int(np.clip(np.interp(val, [-400, 400], [36, 84]), 36, 84))
            pitches.append(pitch)

    start_time = 0.0
    for pitch in pitches:
        note = pretty_midi.Note(
            velocity=random.randint(60, 100),
            pitch=pitch,
            start=start_time,
            end=start_time + note_length
        )
        instrument.notes.append(note)
        start_time += note_length

    midi.instruments.append(instrument)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    midi.write(output_file)
    return output_file


def generate_lofi_track_from_synthetic_data(df, row_index=None, base_name='lofi_track'):
    row = df.sample(1).iloc[0] if row_index is None else df.iloc[row_index]

    midi_path = f'generated/{base_name}.mid'
    wav_path = f'generated/{base_name}.wav'
    mp3_path = f'generated/{base_name}.mp3'

    print("🎼 Creating MIDI...")
    synthetic_row_to_midi(row, midi_path)

    print("🎹 Converting MIDI to WAV...")
    convert_midi_to_wav(midi_path, wav_path)

    print("🎧 Adding lo-fi FX...")
    add_lofi_effects(wav_path, mp3_path)

    print(f"✅ Track ready: {mp3_path}")
    return mp3_path