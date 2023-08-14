# Model training

A default run can be launched with the command below:
```
python learning/main.py
```


Running this command assumes that the `-da` option points to a folder which contains annotated graphs (networkx RNA graphs associated with a fingerprint vector).

See `data_processor.py` for details on creating annotated graphs.

Most important model parameters which can be specified from the command line:

* node embedding dimensions
* graph pooling function
* whether to warm start the model with a pre-trained embedder

For a full list of command line options run:

```
python learning/main.py -h
```

All necessary files are saved in a folder inside `results/` with the run id specified at command ine (`-n`)

Once training is complete you can load the model to make new predictions.
