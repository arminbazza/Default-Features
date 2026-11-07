# Default Feature Representations
Code to accompany the paper "Default Feature Representations of the Cognitive Map"

Default Feature Representations (DFR) split a predictive cognitive map into a stable feature basis, shared across environments, and a change operator, learned from experience, that captures how the environment has changed. This mirrors the two-timescale hypothesis of the entorhinal-hippocampal system, in which grid-cell features stay stable across environments while hippocampal maps adapt to each one.

## Usage
### Environment installation
To install the environment make sure to run:
```bash
conda env create -f env.yml
```

### Scripts
All of the scripts for producing the results in the paper can be found in `src/`, all of the figures were made inside of `src/make_figs.ipynb`.