#@title Setup Environment
#@markdown Copy some auxiliary data from Google Cloud Storage.
#@markdown Also install and import Python dependencies needed
#@markdown for running the Transformer models.

print('Copying Salamander piano SoundFont (via https://sites.google.com/site/soundfonts4u) from GCS...')

print('Installing dependencies...')

print('Importing libraries...')

#Sets up environment

import numpy as np
import os
import tensorflow.compat.v1 as tf

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

from magenta.models.score2perf import score2perf
import note_seq

tf.disable_v2_behavior()

print('Done!')

#Create definitions

#@title Definitions
#@markdown Define a few constants and helper functions.

SF2_PATH = './content/Yamaha-C5-Salamander-JNv5.1.sf2'
SAMPLE_RATE = 16000

# Upload a MIDI file and convert to NoteSequence.
def upload_midi():
  #data = list(files.upload().values())
  #if len(data) > 1:
   # print('Multiple files uploaded; using only one.')
  #return note_seq.midi_to_note_sequence(data[0])
  return note_seq.midi_file_to_note_sequence("./content/c_major_arpeggio.mid")

# Decode a list of IDs.
def decode(ids, encoder):
  ids = list(ids)
  if text_encoder.EOS_ID in ids:
    ids = ids[:ids.index(text_encoder.EOS_ID)]
  return encoder.decode(ids)

#@title Setup and Load Checkpoint
#@markdown Set up generation from a melody-conditioned
#@markdown Transformer model.

model_name = 'transformer'
hparams_set = 'transformer_tpu'
ckpt_path = 'gs://magentadata/models/music_transformer/checkpoints/melody_conditioned_model_16.ckpt'

class MelodyToPianoPerformanceProblem(score2perf.AbsoluteMelody2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True

problem = MelodyToPianoPerformanceProblem()
melody_conditioned_encoders = problem.get_feature_encoders()

# Set up HParams.
hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
trainer_lib.add_problem_hparams(hparams, problem)
hparams.num_hidden_layers = 16
hparams.sampling_method = 'random'

# Set up decoding HParams.
decode_hparams = decoding.decode_hparams()
decode_hparams.alpha = 0.0
decode_hparams.beam_size = 1

# Create Estimator.
run_config = trainer_lib.create_run_config(hparams)
estimator = trainer_lib.create_estimator(
    model_name, hparams, run_config,
    decode_hparams=decode_hparams)

# These values will be changed by the following cell.
inputs = []
decode_length = 0

# Create input generator.
def input_generator():
  global inputs
  while True:
    yield {
        'inputs': np.array([[inputs]], dtype=np.int32),
        'targets': np.zeros([1, 0], dtype=np.int32),
        'decode_length': np.array(decode_length, dtype=np.int32)
    }

# Start the Estimator, loading from the specified checkpoint.
input_fn = decoding.make_input_fn_from_generator(input_generator())
melody_conditioned_samples = estimator.predict(
    input_fn, checkpoint_path=ckpt_path)

# "Burn" one.
_ = next(melody_conditioned_samples)

#@title Choose Melody
#@markdown Here you can choose a melody to be accompanied by the
#@markdown model.  We have provided a few, or you can upload a
#@markdown MIDI file; if your MIDI file is polyphonic, the notes
#@markdown with highest pitch will be used as the melody.

# Tokens to insert between melody events.
event_padding = 2 * [note_seq.MELODY_NO_EVENT]

'''
melodies = {
    'Mary Had a Little Lamb': [
        64, 62, 60, 62, 64, 64, 64, note_seq.MELODY_NO_EVENT,
        62, 62, 62, note_seq.MELODY_NO_EVENT,
        64, 67, 67, note_seq.MELODY_NO_EVENT,
        64, 62, 60, 62, 64, 64, 64, 64,
        62, 62, 64, 62, 60, note_seq.MELODY_NO_EVENT,
        note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT
    ],
    'Row Row Row Your Boat': [
        60, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        60, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        60, note_seq.MELODY_NO_EVENT, 62,
        64, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        64, note_seq.MELODY_NO_EVENT, 62,
        64, note_seq.MELODY_NO_EVENT, 65,
        67, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        72, 72, 72, 67, 67, 67, 64, 64, 64, 60, 60, 60,
        67, note_seq.MELODY_NO_EVENT, 65,
        64, note_seq.MELODY_NO_EVENT, 62,
        60, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT
    ],
    'Twinkle Twinkle Little Star': [
        60, 60, 67, 67, 69, 69, 67, note_seq.MELODY_NO_EVENT,
        65, 65, 64, 64, 62, 62, 60, note_seq.MELODY_NO_EVENT,
        67, 67, 65, 65, 64, 64, 62, note_seq.MELODY_NO_EVENT,
        67, 67, 65, 65, 64, 64, 62, note_seq.MELODY_NO_EVENT,
        60, 60, 67, 67, 69, 69, 67, note_seq.MELODY_NO_EVENT,
        65, 65, 64, 64, 62, 62, 60, note_seq.MELODY_NO_EVENT        
    ]
} '''

melody = 'Upload your own!'  #@param ['Mary Had a Little Lamb', 'Row Row Row Your Boat', 'Twinkle Twinkle Little Star', 'Upload your own!']

if melody == 'Upload your own!':
  # Extract melody from user-uploaded MIDI file.
  melody_ns = upload_midi()
  melody_instrument = note_seq.infer_melody_for_sequence(melody_ns)
  notes = [note for note in melody_ns.notes
           if note.instrument == melody_instrument]
  del melody_ns.notes[:]
  melody_ns.notes.extend(
      sorted(notes, key=lambda note: note.start_time))
  for i in range(len(melody_ns.notes) - 1):
    melody_ns.notes[i].end_time = melody_ns.notes[i + 1].start_time
  inputs = melody_conditioned_encoders['inputs'].encode_note_sequence(
      melody_ns)
else:
  # Use one of the provided melodies.
  events = [event + 12 if event != note_seq.MELODY_NO_EVENT else event
            for e in melodies[melody]
            for event in [e] + event_padding]
  inputs = melody_conditioned_encoders['inputs'].encode(
      ' '.join(str(e) for e in events))
  melody_ns = note_seq.Melody(events).to_sequence(qpm=150)

# Play and plot the melody.
'''note_seq.play_sequence(
    melody_ns,
    synth=note_seq.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
note_seq.plot_sequence(melody_ns)'''


#@title Generate Accompaniment for Melody
#@markdown Generate a piano performance consisting of the chosen
#@markdown melody plus accompaniment.

# Generate sample events.
decode_length = 4096
sample_ids = next(melody_conditioned_samples)['outputs']

# Decode to NoteSequence.
midi_filename = decode(
    sample_ids,
    encoder=melody_conditioned_encoders['targets'])
accompaniment_ns = note_seq.midi_file_to_note_sequence(midi_filename)

# Play and plot. 
'''note_seq.play_sequence(
    accompaniment_ns,
    synth=note_seq.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
note_seq.plot_sequence(accompaniment_ns) '''

#@title Download Accompaniment as MIDI
#@markdown Download accompaniment performance as MIDI (optional).

note_seq.sequence_proto_to_midi_file(
    accompaniment_ns, '/tmp/accompaniment.mid')
#files.download('/tmp/accompaniment.mid')
  

