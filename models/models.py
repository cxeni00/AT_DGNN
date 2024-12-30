import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import *

_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PowerLayer(nn.Module):
    """
    The power layer: calculates the log-transformed power of the data
    """

    def __init__(self, dim, length, step):              # 처리할 차원 / 롤링 윈도우 크기 / 폴링 이동 간격
        super(PowerLayer, self).__init__()
        self.dim = dim
        # 평균 폴링 레이어 정의(2D 입력을 대상으로 함)
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))            # 데이터 제곱 -> 평균 폴링 -> 로그 변환

class Aggregator():
    """
    영역별로 데이터 집계
    """

    def __init__(self, idx_area):                    
        self.chan_in_area = idx_area                # 각 영역에 포함된 채널 수
        self.idx = self.get_idx(idx_area)           # 채널을 영역별로 구분하는 인덱스 생성
        self.area = len(idx_area)                   # 총 영역의 수

    def forward(self, x):
        # x: batch x channel x data
        data = []                                               # 각 영역의 집계 데이터를 저장한 리스트
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                # 현재 영역의 데이터를 추출하여 집계
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                # 마지막 영역의 데이터를 추출하여 집계
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)                     # 각 영역 데이터를 랍쳐 새로운 차원 생성

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area                # 각 영역의 시작 인덱스
        idx_ = [0]  
        for i in idx:
            idx_.append(idx_[-1] + i)           # 이전 인덱스에 채널 수를 덯해 다음 시작점 계산
        return idx_[1:]                         # 첫 번째 항목을 제외하고 반환

    def aggr_fun(self, x, dim):
        return torch.mean(x, dim=dim)           # 편규 계산

class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):               # 입력 특징 차원/ 츨략특징 차원 / 편향 사용 여부
        super(GraphConvolution, self).__init__()
        self.in_features = in_features                                      # 입력 특징 차원
        self.out_features = out_features                                    # 출력 특징 차원
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))    # 가중치 초기화
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        
        # 편향 초기화
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):   # 입력 특징 행렬(batch, nede, feature) 인접행렬(batch, node, node)
        output = torch.matmul(x, self.weight) - self.bias
        output = F.relu(torch.matmul(adj, output))
        return output