# Graph WaveNet for Deep Spatial-Temporal Graph Modeling

This is the original pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121).  A nice improvement over GraphWavenet is presented by Shleifer et al. [paper](https://arxiv.org/abs/1912.07390) [code](https://github.com/sshleifer/Graph-WaveNet).



<p align="center">
  <img width="350" height="400" src=./fig/model.png>
</p>

---

논문에 따르면 32 layer에서 40 layer로 확장하면 학습 성능은 5% 증가하는데 비해 학습 파라미터의 수는 54% 증가한다고 하였습니다.

따라서 논문에서 제시한 모델의 layer 개수와 동일한 32 layer로 모델 학습을 진행하였습니다.

### dilated TCN: 시간 의존성 포착

<img width="763" alt="Screen Shot 2024-05-30 at 2 26 07 PM" src="https://github.com/AI-based-ETA/pretrained_AI_Model/assets/65798779/e656e7b5-44d4-42d0-92a0-0869f8112d96">

dilated CNN은 긴 길이의 시계열 데이터를 효과적으로 처리할 수 있습니다.

큰 receptive field를 취하려면, 파라미터의 개수가 많아야 하지만 dilated convolution을 사용하면 receptive field는 커지지만 파라미터의 개수는 늘어나지 않기 때문에 연산량 관점에서 탁월한 효과를 얻을 수 있습니다.

### GCN: 공간 의존성 포착

<img width="787" alt="Screen Shot 2024-05-30 at 2 48 50 PM" src="https://github.com/AI-based-ETA/pretrained_AI_Model/assets/65798779/ed0f646f-0435-4766-a034-5e04402a8bc9">

그래프 기반 컨볼루션 연산을 활용하여 교통량 데이터에서 공간 의존성을 캡처한다. 교툥량 센서의 물리적 위치에 기초하여 그래프를 구성하고, 간선은 센서 간의 쌍방향 거리를 나타낸냅니다. 이러한 연산자를 적용함으로써 컨볼루션 계층은 네트워크 전반에 걸친 교통량 정보의 확산을 효과적으로 모델링한다.



# Performance

한국도로교통공사 공공데이터포탈의 VDS 1.5만개의 데이터의 70%, 10%, 20%로 나눠서 각각 train, validation, test로 사용하였습니다.

## 학습 성능

<img width="500" alt="MAE" src="https://github.com/AI-based-ETA/pretrained_AI_Model/assets/65798779/8fe3047d-3539-408b-a777-e73037372183">
<img width="500" alt="MAPE" src="https://github.com/AI-based-ETA/pretrained_AI_Model/assets/65798779/b11cfe74-136c-49f7-bf01-c8a92d8c55cf">
<img width="500" alt="RMSE" src="https://github.com/AI-based-ETA/pretrained_AI_Model/assets/65798779/bec9dced-112f-4b39-b9ed-6735e18ff183">

---

# 추론 성능

<img width="500" alt="MAE" src="https://github.com/AI-based-ETA/pretrained_AI_Model/assets/65798779/1b072a05-2b95-49e9-b5e2-a0cb9e2917ce">
<img width="500" alt="MAPE" src="https://github.com/AI-based-ETA/pretrained_AI_Model/assets/65798779/f041eb5f-8d8f-4937-af5f-1d9dc0670665">
<img width="500" alt="RMSE" src="https://github.com/AI-based-ETA/pretrained_AI_Model/assets/65798779/55c2bdd1-e3db-48f1-a835-f5cc4118089d">


<table> <thead> <tr><th>예측 시간(분)</th><th>Test Loss</th><th>Test MAPE</th><th>Test RMSE</th></tr></thead>
    <tbody>
        <tr><td>15</td><td>1.7689</td><td>0.0203</td><td>2.9916</td></tr>
        <tr><td>30</td><td>2.0288</td><td>0.0238</td><td>3.5879</td></tr>
        <tr><td>60</td><td>2.3372</td><td>0.0278</td><td>4.2073</td></tr>
    </tbody>
</table>
