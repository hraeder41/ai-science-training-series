## MNIST Training on Graphcore

### Default Parameters:
```
TrainingModelWithLoss(
  (model): Network(
    (layer1): Block(
      (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer2): Block(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer3): Linear(in_features=1600, out_features=128, bias=True)
    (layer3_act): ReLU()
    (layer3_dropout): Dropout(p=0.5, inplace=False)
    (layer4): Linear(in_features=128, out_features=10, bias=True)
    (softmax): Softmax(dim=1)
  )
  (loss): CrossEntropyLoss()
)
Accuracy on test set: 96.66%

Epochs: 100%|██████████| 10/10 [01:56<00:00, 11.61s/it]
  0%|          | 0/125 [00:00<?, ?it/s]                2024-11-14T22:10:36.481589Z PL:POPLIN    2150259.2150259 W: poplin::preplanConvolution() is deprecated! Use poplin::preplan() instead   | 4/100 [00:00<00:19]
                                                       2024-11-14T22:10:38.768289Z PL:POPLIN    2150259.2150259 W: poplin::preplanMatMuls() is deprecated! Use poplin::preplan() instead       | 6/100 [00:02<00:44]
Graph compilation: 100%|██████████| 100/100 [00:15<00:00]
```
### Increase Batch Size to 16
```
TrainingModelWithLoss(
  (model): Network(
    (layer1): Block(
      (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer2): Block(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer3): Linear(in_features=1600, out_features=128, bias=True)
    (layer3_act): ReLU()
    (layer3_dropout): Dropout(p=0.5, inplace=False)
    (layer4): Linear(in_features=128, out_features=10, bias=True)
    (softmax): Softmax(dim=1)
  )
  (loss): CrossEntropyLoss()
)
Accuracy on test set: 98.49%

Epochs: 100%|██████████| 10/10 [01:27<00:00,  8.78s/it]
Graph compilation: 100%|██████████| 100/100 [00:01<00:00]
```
### Decrease Learning Rate to 0.01
```
▒TrainingModelWithLoss(
  (model): Network(
    (layer1): Block(
      (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer2): Block(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer3): Linear(in_features=1600, out_features=128, bias=True)
    (layer3_act): ReLU()
    (layer3_dropout): Dropout(p=0.5, inplace=False)
    (layer4): Linear(in_features=128, out_features=10, bias=True)
    (softmax): Softmax(dim=1)
  )
  (loss): CrossEntropyLoss()
)
Accuracy on test set: 90.10%

Epochs: 100%|██████████| 10/10 [01:27<00:00,  8.73s/it]
Graph compilation: 100%|██████████| 100/100 [00:00<00:00]
```
### Doubling the number of epochs to 20
```
TrainingModelWithLoss(
  (model): Network(
    (layer1): Block(
      (conv): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer2): Block(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (relu): ReLU()
    )
    (layer3): Linear(in_features=1600, out_features=128, bias=True)
    (layer3_act): ReLU()
    (layer3_dropout): Dropout(p=0.5, inplace=False)
    (layer4): Linear(in_features=128, out_features=10, bias=True)
    (softmax): Softmax(dim=1)
  )
  (loss): CrossEntropyLoss()
)

Accuracy on test set: 21.19%

Epochs: 100%|██████████| 20/20 [02:52<00:00,  8.60s/it]
Graph compilation: 100%|██████████| 100/100 [00:00<00:00]
```

### Observations

It appears that increasing the batch size also increased the ability of our model to accurately predict on the MNIST dataset. However, this is a relatively small difference in test accuracy, so I would want to repeat the
prediction process multiple times with each model to see if there is a significant difference. When decreasing the learning rate, we saw a distinct decrease in test accuracy. This is likely because we found a sort of
"local minimum" in the gradient of our loss function, and our learning rate was not large enough to escape it. Finally, when increasing the number of epochs, we actually saw a large decrease in test accuracy as well.
During training, I saw the accuracy increasing as normal, however around 16 epochs in I saw a massive drop in accuracy. This leads me to believe that our model started drastically overfitting the training data, which
led to a significant drop in test accuracy.
