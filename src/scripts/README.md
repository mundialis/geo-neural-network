# mundialis geo-neural-network library

Wrapper scripts to use train, test, and inference functions of the geo-neural-network library.

The scripts require as single argument a config file, examples are in the subfolder [scripts](./scripts).
The purpose of the config file mechanism is that these config files provide documentation on the exact settings used.
Usage of the scripts is thus fairly simple:

1. create config files, best: modify the example config files
2. call the scripts with e.g. `python3 train_geosmp.py /path/to/config_train.ini
