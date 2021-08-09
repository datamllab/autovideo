# Benchmarks

## Hyperparameters to obtain the results
| Hyperparameters   | TSN     | TSM     | ECO                     | I3D   | R3D    | R2P1D         |
| :---------------: | :-----: | :-----: | :---------------------: | :---: | :----: | :-----------: |
| Number of Segments| 16      | 16      | 8                       | 16    | 32     | 32            |
| Epochs            | 50      | 50      | 100                     | 50    | 80     | 80            |
| Weight Decay      | 5e-4    | 5e-4    | 5e-4                    | 5e-4  | 5e-4   | 5e-4          |
| Learning Rate     | 0.001   | 0.0001  | 0.0001                  | 0.01  | 0.001  | 0.001         |
| Batch Size        | 4       | 4       | 8                       | 20    | 16     | 16            |
| Momentum          | 0.9     | 0.9     | (Adam)                  | 0.9   | 0.9    | 0.9           |
| Modality          | RGB     | RGB     | RGB                     | RGB   | RGB    | RGB           |
| Dropout           | 0.8     | 0.5     | 0.8                     | 0.5   | 0.5    | 0.5           |
| Backbone          |ResNet-50|ResNet-50| BN-Inception, ResNet-18 | Inception  |ResNet-50|ResNet-34|

## Results
| Datasets   | TSN     | TSM    | ECO       | I3D   | R3D    | R2P1D   |
| :--------: | :-----: | :----: | :-------: | :---: | :----: | :-----: |
| HMDB51-1   | 48.6%   | 58.17% | 44.4%     | 48.43%| 54.77% | 60.26%  |
| HMDB51-2   | 50.3%   | 55%    | 44.8%     | 47.97%| 52.02% | 59.01%  |
| HMDB51-3   | 48.76%  | 57.65% | 41.5%     | 45.56%| 52.3%  | 59.9%   |
| UCF        | 81.97%  | 88.47% | 77.11%    | 75.51%| 83.74% | 86.92%  |

