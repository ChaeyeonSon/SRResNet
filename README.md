# SRResNet
SRResNet 변형 구조 구현

![image](https://user-images.githubusercontent.com/55138488/86691709-099fd400-c044-11ea-97e9-fa8d8bd819a4.png)

Dataset : DIV2K 데이터
- LR: bicubic downsampling x2
- Train data: 2K 해상도 데이터 800 장
- Validation data: 2K 해상도 데이터 100 장


Architecture
 - All conv filters are of size 3x3 (stride 1)
 - All output channels 64 unless specified below the layer
 - Res block # = 16
 
Training (SRResNet 논문 참고)
 - Adam optimizer, L2 loss
 - Training patch size: 64x64 (LR), mini-batch size: 8
 - Data normalized to [0, 1]
 - Learning rate: 1e-4, Iterations: 1e5
 
Result

Ablation Study
 - Resblock 개수 변화 (4, 8, 12, 16, 24)
 - global connection 이 없는 경우
 - pixel shuffle 대신 bicubic upsampling 사용
 - VGG54 Loss만 content loss 로써 L2 대신 사용
