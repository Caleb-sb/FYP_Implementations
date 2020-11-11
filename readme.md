# EEE4022 Final Year Project
## Active Noise Control

Here, you'll find the code for implementing the various versions of my final year project. All iteration testing can be run from testing.py running it as the main file.

## Requirements

There are several necessary requirements for this project.

```bash
Numpy
Matplotlib
Jack-client
Sklearn
Scipy
Numba
PyAudio
Time
Threading
Sounddevice
```

## Usage

In order to run the tests, use the following bash commands from the root project directory:

For iteration1 tests

```bash
python3 -m testing 1
```

For iteration 2 tests once the required audio devices have been connected and the port numbers queried with sounddevice
```bash
python3 -m testing 2 <duration_of_recording>
```

In order to run iteration 3, once the stereo audio input device has been connected and both are being used by your JACK audio server:
```bash
python3 -m testing 3
```
Note: Iteration 3 is currently causing an unstable adaptive filter, most likely due to an indexing issue or misaligned audio streams from JACK. Run with caution if using headphones as the resulting sound output volume is high.
