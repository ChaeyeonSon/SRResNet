# SRResNet
SRResNet 변형 구조 구현

![image](https://user-images.githubusercontent.com/55138488/86691709-099fd400-c044-11ea-97e9-fa8d8bd819a4.png)

## Dataset : DIV2K 데이터
- LR: bicubic downsampling x2
- Train data: 2K 해상도 데이터 800 장
- Validation data: 2K 해상도 데이터 100 장


## Architecture
 - All conv filters are of size 3x3 (stride 1)
 - All output channels 64 unless specified below the layer
 - Res block # = 16
 
## Training (SRResNet 논문 참고)
 - Adam optimizer, L2 loss
 - Training patch size: 64x64 (LR), mini-batch size: 8
 - Data normalized to [0, 1]
 - Learning rate: 1e-4, Iterations: 1e5
 
## Result
<img src="https://user-images.githubusercontent.com/55138488/87017247-b926a300-c20a-11ea-8135-a2e8576caaad.png" width="30%"> <img src="https://user-images.githubusercontent.com/55138488/87017325-ccd20980-c20a-11ea-8a0a-fbff2215b996.png" width="30%">  
Avg. Loss : 0.00063688, Avg. PSNR : 34.2719, Avg. SSIM : 0.9311, Avg. BICUBIC_PSNR : 31.2743, Avg. BICUBIC_SSIM : 0.9061  
*LR* <img src = "https://user-images.githubusercontent.com/55138488/87021592-5f28dc00-c210-11ea-9998-8700a7c15f3d.jpg" width="20%">
*HR* <img src = "https://user-images.githubusercontent.com/55138488/87021634-694ada80-c210-11ea-9165-2c4ed3252d3a.jpg" width="20%">
*Bicubic* <img src = "https://user-images.githubusercontent.com/55138488/87021857-ad3ddf80-c210-11ea-8000-3c0617e0f72a.jpg" width="20%">
*Pred* <img src = "https://user-images.githubusercontent.com/55138488/87022149-07d73b80-c211-11ea-97b7-f0e66e96037d.jpg" width="20%">
### Ablation Study Result (img) *(1)* *(2)* *(3)* *(4)* *(5)* *(6)*
<img src = "https://user-images.githubusercontent.com/55138488/87024523-31de2d00-c214-11ea-99de-ff2e7412374d.jpg" width="20%"> <img src = "https://user-images.githubusercontent.com/55138488/87024527-330f5a00-c214-11ea-8b51-de069c1b6ea2.jpg" width="20%"> <img src = "https://user-images.githubusercontent.com/55138488/87024530-330f5a00-c214-11ea-9796-844bdd8e6f18.jpg" width="20%">  

<img src = "https://user-images.githubusercontent.com/55138488/87024534-34408700-c214-11ea-9dcf-4a8b66153156.jpg" width="20%"> <img src = "https://user-images.githubusercontent.com/55138488/87024531-33a7f080-c214-11ea-84f5-42c0338c47b1.jpg" width="20%"> <img src = "https://user-images.githubusercontent.com/55138488/87024533-33a7f080-c214-11ea-953c-97adc4402c82.jpg" width="20%">

## Ablation Study
 - Resblock 개수 변화 (8, 12, 16=original, 24) 
   - (1) 08개 - Avg. Loss : 0.00066879, Avg. PSNR : 33.9920, Avg. SSIM : 0.9267  
   <img src="https://user-images.githubusercontent.com/55138488/87023502-bf208200-c212-11ea-9d54-988f26fe1ecd.png" width="30%"> <img src="https://user-images.githubusercontent.com/55138488/87023571-d9f2f680-c212-11ea-9481-ed9f61916cb8.png" width="30%">  
   - (2) 12개 - Avg. Loss : 0.00064017, Avg. PSNR : 34.2200, Avg. SSIM : 0.9338  
   <img src="https://user-images.githubusercontent.com/55138488/87024020-8c2abe00-c213-11ea-8ddf-0427e3e3ea26.png" width="30%"> <img src="https://user-images.githubusercontent.com/55138488/87024117-a9f82300-c213-11ea-99ed-efe78e33139f.png" width="30%">
   - (3) 24개 - Avg. Loss : 0.00064198, Avg. PSNR : 34.2591, Avg. SSIM : 0.9337  
   <img src="https://user-images.githubusercontent.com/55138488/87024225-cd22d280-c213-11ea-95f3-6740b6da981e.png" width="30%"> <img src="https://user-images.githubusercontent.com/55138488/87024174-bc725c80-c213-11ea-8392-a9acfda2a608.png" width="30%">  
   - training 결과는 기존 모델까지 4개 모두 비슷했지만 validation 에서 8개는 확연히 성능이 떨어졌으며 나머지 3개는 수치 자체는 비슷했다.  
   - 12, 16, 24 에서 12개가 가장 안정적인 valid graph를 보여주었고, 24개가 너무 block 개수가 많은 것인지 매우 요동치는 valid graph 를 보여주었다.  
   - 12개가 가장 best라고 생각하며 차후 실험들에서는 12개 block 으로 실험을 진행했다.  
   
   
 *여기부터 block 12개 사용, 남색 그래프 = block#이 12인 SRResnet*  
 - (4) global connection 이 없는 경우 - Avg. Loss : 0.00073284, Avg. PSNR : 33.1214, Avg. SSIM : 0.9360   
 <img src="https://user-images.githubusercontent.com/55138488/87024816-900b1000-c214-11ea-9d7b-00909f1ffb07.png" width="30%"> <img src="https://user-images.githubusercontent.com/55138488/87025009-c6e12600-c214-11ea-82f5-3db708314e94.png" width="30%">
    - training loss 수렴 정도는 비슷하지만 수렴 속도가 global connection 이 있는 경우가 훨씬 빠른 것을 볼 수 있다.  
    - 또한 valid 역시 global connection이 있는 경우보다 약간 떨어진다.  
    
    
 - (5) pixel shuffle 대신 bicubic upsampling 사용 - Avg. Loss : 0.00197655, Avg. PSNR : 34.0176, Avg. SSIM : 0.9329    
 <img src="https://user-images.githubusercontent.com/55138488/87025103-ec6e2f80-c214-11ea-9b23-6f4b47389f27.png" width="30%"> <img src="https://user-images.githubusercontent.com/55138488/87025153-fbed7880-c214-11ea-8b37-25acb1dc4047.png" width="30%">  
   - 사실상 수치는 약간 떨어지지만 기존 방식과 성능이 거의 비슷하다.  
   - 무엇이 문제였는지 좀 더 생각해보아야 할 듯 하다.  
   
   
 - (6) VGG54 Loss만 content loss 로써 L2 대신 사용 - Avg. Loss : 0.01088041, Avg. PSNR : 24.9014, Avg. SSIM : 0.5738  
 loss 는 vgg loss 를 사용했으므로 기존 mse 와 비교 불가    
 <img src="https://user-images.githubusercontent.com/55138488/87025303-2c351700-c215-11ea-9451-98bc65ed0eef.png" width="30%"> <img src="https://user-images.githubusercontent.com/55138488/87025434-5dade280-c215-11ea-94e4-2afad3d8eb82.png" width="30%">  
   - GAN 없이 vgg loss만 content loss 로 사용하는 것은 무리였다.  
   - mse 와 같이 사용해보면 또 어떨까? 의미가 없을까?  
   - 사진을 더 확대해 보면 약간 줄무늬가 나타나지만 그냥 큰 사진으로보면 육안으로는 생각보다 우수하다.  
   - 하지만 수치면에서 다른 모델들보다 결과가 매우 열악한 것이 눈에 띈다.
 
