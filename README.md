# Breast Cancer Classification with a Small PyTorch Network

A minimal feed-forward neural network that classifies breast tumors as malignant or benign from 30 numeric measurements. The project is a learning exercise: every step of a standard deep learning pipeline is written out explicitly (data prep, model, loss, optimizer, training loop, evaluation) with no high-level training wrappers.

## Dataset

Breast Cancer Wisconsin (Diagnostic), loaded through `sklearn.datasets.load_breast_cancer`.

- 569 samples, 30 numeric features per sample
- Features are computed from digitized images of fine needle aspirates of breast masses (radius, texture, perimeter, and so on; mean, standard error and worst value of each)
- Target encoding in scikit-learn: `0` = malignant, `1` = benign (212 malignant, 357 benign)

## Model

| Layer | Shape | Activation |
|---|---|---|
| Input | 30 | none |
| Hidden 1 (`fc1`) | 30 to 64 | ReLU |
| Hidden 2 (`fc2`) | 64 to 32 | ReLU |
| Output (`fc3`) | 32 to 1 | Sigmoid |

4,097 trainable parameters. The output is the predicted probability that the tumor is benign (class `1`).

## Training setup

| Setting | Value |
|---|---|
| Split | 80% train (455) / 20% test (114) |
| Preprocessing | `StandardScaler`, fitted on the training set only |
| Loss | Binary cross-entropy (`nn.BCELoss`) |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 32, shuffled each epoch |
| Epochs | 20 |
| Decision threshold | 0.5 |

## Results

From the recorded run in the notebook:

| Metric | Value |
|---|---|
| Final training loss (epoch 20) | 0.0322 |
| Test loss | 0.0626 |
| Test accuracy | 0.9825 (112 of 114 correct) |

The split is not seeded, so a fresh run will give slightly different numbers. With only 114 test samples, one extra mistake moves accuracy by about 0.9 percentage points. Treat the figure as a rough indication, not a stable estimate.

## Run it

```bash
pip install torch scikit-learn jupyter
jupyter notebook
```

Open the notebook and run all cells from top to bottom. The notebook was developed on Python 3.12.

## Limitations

- Single random split, no validation set, no cross-validation
- Accuracy only. In a screening context, sensitivity (missed malignancies) and specificity matter more than overall accuracy, so a confusion matrix and per-class metrics would be the natural next addition
- Tabular features extracted beforehand, not raw images, and a small, well-studied benchmark dataset. The result says nothing about clinical performance
- Not a medical device and not for diagnostic use

## Further reading

See `Notes.md` for a step-by-step explanation of every part of the notebook, written for readers new to deep learning.

## References

- Street, W. N., Wolberg, W. H., Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE International Symposium on Electronic Imaging*, vol. 1905. (Original dataset paper; verify volume and page details before citing formally.)
- UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu
- scikit-learn documentation, `load_breast_cancer`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- Kingma, D. P., Ba, J. (2015). Adam: A method for stochastic optimization. ICLR 2015. arXiv:1412.6980