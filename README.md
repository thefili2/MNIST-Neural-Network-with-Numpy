# NumPy Neural Network: No PyTorch, No Tensorflow

Most ML tutorials slap together a few `keras.layers`, hit `.fit()`, and call it a day. This is the opposite of that.

This is a fully functional Multi-Layer Perceptron (MLP) built entirely from scratch in raw NumPy. Backpropagation, gradient descent, batch normalization, and the Adam optimizer, all hand-coded matrix math. Zero deep learning frameworks used for the actual logic.

## How does the code work:

- **Pure NumPy:** No PyTorch, no TensorFlow, no JAX. Just matrices, dot products. (TensorFlow is imported only to grab the MNIST dataset because parsing raw IDX files is a headache nobody needs).
- **Hand-rolled Layers:** Custom Dense, ReLU, Dropout, and BatchNorm classes. Each has its own forward and backward pass.
- **Adam Optimizer:** Implemented from scratch, including bias correction and He initialization to keep the ReLUs from dying.
- **Inverted Dropout:** Scales weights during training so inference requires no extra steps.

## The Architecture

A pretty standard deep MLP bottlenecking down to 10 classes:

```text
Input (784)
  ├─ Dense(256) -> BatchNorm -> ReLU -> Dropout(0.2)
  ├─ Dense(128) -> BatchNorm -> ReLU -> Dropout(0.2)
  ├─ Dense(64)  -> BatchNorm -> ReLU -> Dropout(0.2)
  └─ Dense(10)  -> Softmax
```

## How to run it

You need NumPy and TensorFlow (just for the data loader).

```bash
pip install numpy tensorflow
python main.py
```

## Expected Output

It trains for 25 epochs with a batch size of 64. You should see it climb to around **97.5% - 98%** test accuracy fairly quickly (DEPENDS ON YOUR HARDWARE!!!).

```text
Epoch 1/25  | Loss: 0.5421 | Test Acc: 0.9512
Epoch 2/25  | Loss: 0.2143 | Test Acc: 0.9648
...
Epoch 25/25 | Loss: 0.0312 | Test Acc: 0.9801

Final Accuracy: 98.01%
```

## Why bother?

Because calling `.backward()` in PyTorch feels like magic until you actually have to derive the chain rule for a batch normalization layer yourself.
