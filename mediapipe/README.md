# Mediapipe

Examples of Python scripts using [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide), mostly adapted from [the example repo](https://github.com/google-ai-edge/mediapipe-samples) (and GPT...).

## Requirements

Except for the llm inference examples (in JavaScript, requiring only to download the model and launch a web server), to start playing with this part of the repo, the simplest way is to create an environment and install mediapipe, for instance:

```bash
conda create -n mediapipe python
conda activate mediapipe
(mediapipe) which pip # verify we are in the right env
(mediapipe) pip install mediapipe
```

Voilà!

Then for instance:

```bash
(mediapipe) cd pose_landmarker/
(mediapipe) python pose_landmarker_webcam.py
```

The scripts are designed to download the model they need if they don't find it in the directory.

## Note

I like to keep my Python code tidy, and using external programs can help you with that. I recommend [ruff](https://docs.astral.sh/ruff/) ([black](https://github.com/psf/black) is great, too), which can be installed easily:

```bash
(mediapipe) pip install ruff
```

Then you can do: `ruff check` to check for syntax, and `ruff format` for formatting.
