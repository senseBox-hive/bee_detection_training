# Model training and quantization

requires a `../data` directory (relative to repository)
`../data` should contain

`train/`, `val/`, `test/`
containing `background`, `bee_motion`, `bee_slow` each

## Evaluation results

quantization requires esp-ppq
```
pip install git+https://github.com/espressif/esp-ppq.git
```

source model (as of current commit)
```
Accuracy: 0.9394
Confusion Matrix (rows: actual, columns: predicted):
            background  bee_motion    bee_slow
background        1152          30          18
bee_motion          26         880          32
  bee_slow          12          36         357
```

quantized model (as of current commit)
```
Quantized model accuracy: 0.9335
Quantized confusion matrix (rows: actual, columns: predicted):
            background  bee_motion    bee_slow
background        1169          22           9
bee_motion          42         864          32
  bee_slow          26          38         341
```