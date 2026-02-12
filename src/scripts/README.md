# mundialis geo-neural-network library

Wrapper scripts to use train, test, and inference functions of the geo-neural-network library.

The scripts require as single argument a config file, examples are in the subfolder [configs](./configs).
The purpose of the config file mechanism is that these config files provide documentation on the exact settings used.
Usage of the scripts is thus fairly simple:

1. create config files, best: modify the example config files
2. call the scripts with e.g. `python3 train_geosmp.py /path/to/config_train.ini`

## Config

### Training

* Optional parameter: `weighted_loss` within the section `model`
  * When not set simply the Jaccard loss is used
  * If set, the weights of different model classes are used for CrossEntropy-loss. This is recommended in case of class imbalance.
  * Then weighted loss combination of Jaccard (0.7) and CrossEntropy (0.3) is used (Note: combination weighting currently hardcoded; if useful, can be set as parameter)
  * Hint: for setting `weighted_loss` of model classes, following could help
    ```python
    # get amount of data for each class
    from collections import Counter
    class_counts = Counter()
    # Note: need dataset without augmentation
    # e.g. with train_dataset_no_augmentation = GdalImageDataset(x_train_dir,y_train_dir)
    for idx in range(len(train_dataset_no_augmentation)):
        _, mask = train_dataset_no_augmentation[idx]
        class_counts.update(mask.flatten())

    # 'num_classes' should be same as 'out_classes' from config
    num_classes = 4
    counts = list()
    for i in range(num_classes):
       counts.append(class_counts.get(i, 0))
    total_samples = sum(counts)

    # weights dependent of amount/frequency of class
    freq_weights=list()
    for c in counts: freq_weights.append(0 if c == 0 else total_samples / (num_classes * c))

    # weights dependent of importance of class -> Note must be set manually
    # e.g.
    importance = {0: 0, 1: 1, 2: 1, 3: 0.5}
    imp_list=list()
    for i in range(num_classes):
        imp_list.append(importance.get(i, 1.0))

    # final weighted_loss as combination of frequency and importance of class
    final_weights=list()
    for f, i in zip(freq_weights, imp_list):
        final_weights.append(f*i)

    # Note: Weights do not have to add up to 1, PyTorch normalizes internally
    ```
