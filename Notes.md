# Notes: Building a Neural Network from Zero

A walkthrough of the breast cancer notebook for someone who has never done deep learning. Each section says what the code does and why it is there.

---

## 1. The problem

We have 569 tumor samples. For each one, a lab measured 30 numbers (size, shape, texture of the cell nuclei). A doctor already labeled each sample as malignant or benign.

Goal: build a function that takes the 30 numbers and outputs the right label, including for samples it has never seen.

This is **supervised binary classification**. Supervised means we learn from labeled examples. Binary means there are two possible answers.

## 2. What "deep learning" means in one page

A neural network is a function with many adjustable numbers inside it, called **parameters** (or weights). Learning means finding parameter values that make the function give good answers.

The learning loop always has the same four pieces:

1. **Forward pass.** Feed data in, get predictions out.
2. **Loss.** One number that says how wrong the predictions are. Lower is better.
3. **Backward pass.** Compute, for every parameter, which direction would reduce the loss. These directions are called **gradients**. PyTorch computes them automatically (automatic differentiation).
4. **Update.** Nudge each parameter a small step in the direction that reduces the loss.

Repeat many times. That is all of it. The rest of the notebook is the setup needed to run this loop properly.

## 3. Step by step through the notebook

### 3.1 Imports

- `torch`, `torch.nn`, `torch.nn.functional`: PyTorch, the library that builds and trains the network.
- `DataLoader`, `TensorDataset`: tools to feed data to the network in batches.
- `load_breast_cancer`, `train_test_split`, `StandardScaler`: scikit-learn helpers for the data and its preparation.

### 3.2 Loading the data

```python
x, y = load_breast_cancer(return_X_y=True)
```

`x` has shape (569, 30): 569 samples, 30 features each. `y` has shape (569,): one label per sample.

In this dataset `0` means malignant and `1` means benign. This matters later: the network's output will be the probability of **benign**, not of cancer.

### 3.3 Train/test split

```python
train_test_split(x, y, test_size=0.2)
```

We hide 20% of the data (114 samples) and never let the model learn from it. After training, we test on that hidden part. A model can memorize its training data and still fail on new data, so performance on unseen samples is the only number that tells you whether it learned something general.

Note: no `random_state` is set, so each run splits differently and the final accuracy changes a little between runs. Adding `random_state=42` (any fixed number) makes results reproducible. Passing `stratify=y` additionally keeps the malignant/benign ratio equal in both parts.

### 3.4 Standardization

The 30 features live on very different scales. One might be around 0.1, another around 1000. Networks train badly when inputs differ that much in size, because the large-valued features dominate the arithmetic.

`StandardScaler` fixes this. For each feature it subtracts the mean and divides by the standard deviation, so every feature ends up with mean 0 and standard deviation 1.

(This is not squeezing values into 0 to 1. That would be `MinMaxScaler`. Standardized values can be negative or larger than 1.)

Two calls, two different jobs:

- `fit_transform(x_train)` learns the mean and standard deviation from the training data, then applies them.
- `transform(x_test)` applies the **training** mean and standard deviation to the test data, without learning new ones.

Why not fit on the test data too? Because the test set plays the role of future, unseen data. Using its statistics would leak information from it into the pipeline and make the test result look better than it deserves. This mistake is called **data leakage**.

### 3.5 Converting to tensors

```python
torch.from_numpy(x_train_scaled).float()
```

PyTorch works with **tensors**, its version of NumPy arrays that can also track gradients and run on a GPU. `.float()` converts to 32-bit floating point, which is what the network layers expect.

```python
torch.from_numpy(y_train).float().unsqueeze(1)
```

The labels start with shape (455,). The network outputs shape (455, 1), one column per sample. `unsqueeze(1)` adds that second dimension so the loss function compares like with like.

The notebook prints the shapes to check: (455, 30), (455, 1), (114, 30), (114, 1). Checking shapes early is a good habit, because shape mismatches cause most beginner bugs.

### 3.6 Dataset and DataLoader

```python
train_dataset = TensorDataset(x_train_scaled_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

`TensorDataset` pairs each sample with its label. `DataLoader` hands them out in **batches** of 32.

Why batches instead of all 455 at once, or one at a time?

- One sample at a time gives noisy, slow updates.
- All at once is expensive on large datasets and gives few updates per pass.
- A batch is a compromise. It gives a reasonably stable gradient estimate and many updates per pass.

455 samples at batch size 32 gives 15 batches per pass: 14 full batches and one last batch of 7.

`shuffle=True` reorders the data every pass, so the model does not learn anything from the order of the samples.

### 3.7 The model

```python
class BCNet(nn.Module):
    def __init__(self):
        super(BCNet, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
```

`nn.Linear(30, 64)` is a **fully connected layer**. It takes 30 numbers in and produces 64 numbers out. Each of the 64 outputs is a weighted sum of all 30 inputs plus a constant (bias). That layer holds 30 × 64 weights + 64 biases = 1,984 parameters.

The three layers together:

| Layer | Parameters |
|---|---|
| fc1 (30 to 64) | 1,984 |
| fc2 (64 to 32) | 2,080 |
| fc3 (32 to 1) | 33 |
| Total | 4,097 |

The first two layers are **hidden layers**: their outputs are internal, not the final answer. The sizes 64 and 32 are design choices (hyperparameters). Nothing makes them special, and other sizes would also work on a problem this small.

```python
def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.sigmoid(self.fc3(x))
    return x
```

`forward` defines what happens to the data when it passes through the model.

**Why activation functions?** A weighted sum of weighted sums is still just a weighted sum. Without something in between, stacking layers adds nothing, and the whole network collapses into a single linear model. An **activation function** applied after each layer breaks that linearity and lets the network represent curved, complex relationships.

- **ReLU** is `max(0, x)`: negative values become 0, positive values pass through. It is cheap, simple, and the standard default for hidden layers.
- **Sigmoid** is `1 / (1 + e^(-x))`. It squashes any number into the range 0 to 1, so the output can be read as a probability. It is used on the last layer only, because we want one probability for a yes/no question.

So the final number answers: "how likely is this tumor benign?"

### 3.8 Loss function

```python
criterion = nn.BCELoss()
```

**Binary cross-entropy** measures how wrong a predicted probability `p` is, given the true label `y`:

`loss = -[ y · log(p) + (1 - y) · log(1 - p) ]`

Intuition: if the true label is 1 and the model says 0.99, the loss is tiny. If the model says 0.01, the loss is huge. Confident wrong answers are punished much harder than hesitant ones, which is the behavior you want when training a probability estimator.

### 3.9 Optimizer

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

The optimizer performs the update step. The simplest version, gradient descent, subtracts `learning_rate × gradient` from every parameter. **Adam** is a refinement that adapts the step size for each parameter individually using running averages of past gradients. It is a common default because it works well with little tuning.

`lr=0.001` is the **learning rate**, the overall step size. Too large and training becomes unstable. Too small and it takes forever. 0.001 is Adam's usual starting value.

### 3.10 The training loop

```python
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predication = model(x_batch)
        loss = criterion(predication, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_batch.size(0)
    print(...)
```

One **epoch** is one full pass over the training data. Here: 20 epochs.

Inside, for every batch:

1. `optimizer.zero_grad()` clears the gradients from the previous batch. PyTorch accumulates gradients by default, so skipping this line would silently mix batches together.
2. `model(x_batch)` is the forward pass.
3. `criterion(...)` computes the loss.
4. `loss.backward()` is the backward pass. It fills in a gradient for every parameter.
5. `optimizer.step()` updates the parameters using those gradients.

`model.train()` switches the model into training mode. It makes no difference for this network, but layers like dropout and batch normalization behave differently in training and evaluation, so it is good practice to always set it.

`running_loss` multiplies each batch loss by the batch size before summing. That weights each batch by how many samples it holds, so the smaller final batch does not count too much. Dividing by the dataset size at the end gives the true average loss per sample.

**Reading the output.** The loss falls from 0.6308 to 0.0322. The first value is near 0.69, which is what you get from guessing 0.5 for everything (that is `ln 2`). So the model starts out knowing nothing, and by epoch 20 it fits the training data well. The curve flattens toward the end, meaning most of the learning is already done.

### 3.11 Evaluation

```python
with torch.no_grad():
    model.eval()
    predictions = model(x_test_scaled_tensor)
    loss = criterion(predictions, y_test_tensor).item()
    accuracy = ((predictions >= 0.5) == y_test_tensor).float().mean().item()
```

- `torch.no_grad()` turns off gradient tracking. We are only predicting, not learning, so this saves memory and time.
- `model.eval()` switches to evaluation mode (the counterpart of `model.train()`).
- `predictions >= 0.5` converts probabilities to hard labels: 0.5 or above counts as benign (1), below counts as malignant (0). This cutoff is called the **decision threshold**.
- Comparing to the true labels gives True/False per sample, and the mean of those is the accuracy.

**Result:** test loss 0.0626, test accuracy 0.9825. With 114 test samples, that is 112 correct and 2 wrong.

The test loss is higher than the final training loss (0.0626 against 0.0322). This is normal. Models almost always do slightly better on data they trained on. A very large gap would signal **overfitting**, which means memorizing the training set instead of learning general patterns. Here the gap is small.

## 4. Caveats worth knowing

- **Accuracy is not enough for medical problems.** Missing a malignant tumor (false negative) is a much worse error than flagging a benign one (false positive). A real evaluation would report sensitivity, specificity and a confusion matrix, and might lower the threshold below 0.5 to favor catching malignancies.
- **One split is a noisy estimate.** With 114 test samples, each error moves accuracy by about 0.9 points. Cross-validation or repeated seeded splits give a more trustworthy figure.
- **No validation set.** The number of epochs (20) was fixed in advance. For tuning choices like epochs or layer sizes, you need a third data portion (validation) so the test set stays untouched until the very end.
- **Numerical stability.** PyTorch's documentation recommends `nn.BCEWithLogitsLoss` (sigmoid and loss combined) over a separate sigmoid followed by `nn.BCELoss`, because the combined version is more stable numerically. The current approach works fine here, and this is only a note for later projects. Also check the current docs on whether `F.sigmoid` is deprecated in favor of `torch.sigmoid` in your installed version.
- **This is a benchmark dataset.** Good results here do not transfer to clinical use.

## 5. Glossary

| Term | Meaning |
|---|---|
| Feature | One input measurement (one of the 30 columns) |
| Label / target | The correct answer for a sample |
| Parameter / weight | An adjustable number inside the model |
| Hyperparameter | A choice made by the programmer, not learned (layer sizes, learning rate, epochs, batch size) |
| Tensor | PyTorch's array type |
| Forward pass | Data flows through the model to produce predictions |
| Loss | A single number measuring how wrong the predictions are |
| Gradient | The direction in which a parameter should change to reduce the loss |
| Backpropagation | The algorithm that computes all gradients efficiently (`loss.backward()`) |
| Epoch | One full pass over the training data |
| Batch | The small group of samples used for one update |
| Overfitting | Fitting the training data too closely and generalizing poorly |
| Activation function | A nonlinear function applied between layers |

## 6. References

- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press. Free online at https://www.deeplearningbook.org (backpropagation, activations, optimization, generalization).
- Kingma, D. P., Ba, J. (2015). Adam: A method for stochastic optimization. ICLR 2015. arXiv:1412.6980.
- PyTorch documentation: `nn.Linear`, `nn.BCELoss`, `nn.BCEWithLogitsLoss`, `torch.optim.Adam`, `DataLoader`, https://pytorch.org/docs/stable/index.html
- scikit-learn documentation: `StandardScaler` and `load_breast_cancer`, https://scikit-learn.org/stable/
- Street, W. N., Wolberg, W. H., Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE International Symposium on Electronic Imaging*, vol. 1905. (Verify bibliographic details before formal citation.)
