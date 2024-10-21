# Canvas

P5.js-like Python library (works in Jupyter Notebooks as well)

[Repo](https://github.com/colormotor/py5canvas)

## Installation

With conda, see [here](https://github.com/colormotor/py5canvas?tab=readme-ov-file#installing-dependencies-with-conda).

With pip:
1) create an environment:

```bash
# tested for 3.12 (should work in 3.10, 3.8)
python -m venv env --prompt canvas
```

2) activate it:

```bash
source env/bin/activate
(canvas) $ which pip # should show the path to the `env` directory
```

3) install dependencies:

```bash
(canvas) $ pip install -r requirements.txt
```

## Running sketches

```bash
(canvas) $ python examples/basic_animation.py
```

See also the cool [gui](https://github.com/colormotor/py5canvas?tab=readme-ov-file#gui-support-and-parameters).

```bash
(canvas) $ python examples/parameters.py
```
