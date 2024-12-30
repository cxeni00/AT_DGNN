import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import *
from models.models import GraphConvolution, Aggregator, PowerLayer

_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ATDGNN 클래스 정의
class ATDGNN(nn.Module):

    # tempral_learner : 시간적 특징 학습기 생성
    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
        super(ATDGNN, self).__init__()

        self.num_T = num_T                      # 시간 필터의 출력 채널 수
        self.out_graph = out_graph              # GCN의 출력 특징 수
        self.dropout_rate = dropout_rate        # 드롭 아웃 비율
        self.window = [0.5, 0.25, 0.125]        # 윈도우 사이즈 지정
        self.pool = pool                        # pooling 길이
        self.pool_step_rate = pool_step_rate    # pooling 단계 비울
        self.idx = idx_graph                    # 그래프 인덱스
        self.channel = input_size[1]            # 임력 데이터의 채널 수
        self.brain_area = len(self.idx)         # 뇌 영역 수
        
        self.model_dim = round(num_T / 2)
        self.num_heads = 8

        self.window_size = 100                  # 윈도우 사이즈
        self.stride = 20                        # stride
        
        hidden_features = input_size[2]         # 입력 데이터의 숨겨진 특징 수

        # temporal learner 모듈 설정
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
        # Batch normalization layers
        self.bn_t = nn.BatchNorm2d(num_T)
        self.bn_s = nn.BatchNorm2d(num_T)

        # 1*1 컨볼루션 레이어
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2))
        )
        
        # 특성 통합 및 슬라이딩 윈도우 프로세서 초기화
        self.feature_integrator = FeatureIntegrator(in_channels=32, out_channels=self.model_dim)
        self.sliding_window_processor = SlidingWindowProcessor(model_dim=self.model_dim, num_heads=self.num_heads,
                                                               window_size=self.window_size, stride=self.stride)
        # 로컬 필터 가중치 및 편향 초기화
        size = self.get_size_temporal(input_size)
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)
        # aggregate function
        self.aggregate = Aggregator(self.idx)

        # Dynamic Graph Convolution Layers
        self.dynamic_gcn = StackedDynamicGraphConvolution(size[-1], hidden_features, out_graph, num_layers=3)
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)

        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)

        # Fully connected layer for classification
        self.fc = nn.Sequential(  
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes)
        )

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)

        out = self.feature_integrator(out)  # 특성 통합
        out = self.sliding_window_processor(out)  # 슬라이딩 윈도우 처리

        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def forward(self, x):
        # Temporal convolution
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)

        out = self.feature_integrator(out)  # 특성 통합
        out = self.sliding_window_processor(out)  # 슬라이딩 윈도우 처리

        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)               # Aggregator 적용
        out = self.bn(out)                              
        out = self.dynamic_gcn(out)                     # 그래프 컨볼
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)               # Flatten
        out = self.fc(out)                              # fully connected 레이어 통과
        return out


class DynamicGraphConvolution(GraphConvolution):
    """
    Dynamic Graph Convolution Layer.
    임력 특징에 따라 동적으로 계산된 유사도 기반의 인접 행렬을 활용하여 그래프 컨볼루션 진행
    """

    def __init__(self, in_features, out_features, bias=True):
        # 부모 클래스의 graphconvolution 초기화
        super(DynamicGraphConvolution, self).__init__(in_features, out_features, bias)

    def forward(self, x, adj=None):
        # 인접 행렬이 없은 경우, 입력데이터 기반으로 동적으로 생성
        if adj is None:
            adj = self.normalize_adjacency_matrix(x)

        # 그래프 컨볼루션 연산 수행
        output = torch.matmul(x, self.weight)                   # 입력 특징과 가중치 행렬 곱
        if self.bias is not None:
            output += self.bias                                 # 편향 추가
        output = F.relu(torch.matmul(adj, output))              # 인접 행렬을 적용하고 활성화 함수 
        return output

    # 노드 간 유사도를 계산하여 동적 입접 행렬의 기반으로 사용
    def compute_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1) # 두 번째와 세 번째 축 교환
        s = torch.bmm(x, x_)    # 배치 단위로 행렬 곱 연산 수행
        return s

    # 인접 행격 정규화
    def normalize_adjacency_matrix(self, x):
        # x: b, node, feature
        adj = self.compute_similarity(x)        # b, n, n / 유사도 기잔의 인접 행렬 생성
        num_nodes = adj.shape[-1]
        adj = adj + torch.eye(num_nodes).to(DEVICE)      # 자기 자신과의 연결 추가

        rowsum = torch.sum(adj, dim=-1)                  # 각 행의 합(노드의 차수)
        mask = torch.zeros_like(rowsum)                  # 차수가 0인 노드를 처리
        mask[rowsum == 0] = 1                       
        rowsum += mask
        
        d_inv_sqrt = torch.pow(rowsum, -0.5)             # 차수의 역 제곱근 계싼
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)    # 대각 행렬로 변환
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)     # 정규화된 인접 행렬 생성
        return adj


# 여러개의 DynamicGraphConvolution 레이어를 쌓아 구성
class StackedDynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=3, bias=True):
        super(StackedDynamicGraphConvolution, self).__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(DynamicGraphConvolution(in_features, hidden_features, bias=bias))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(DynamicGraphConvolution(hidden_features, hidden_features, bias=bias))
        # Last layer
        self.layers.append(DynamicGraphConvolution(hidden_features, out_features, bias=bias))

    def forward(self, x, adj=None):
        # 각 레이어에 대해 순차적으로 레이어와 인접 행렬 적용
        for layer in self.layers:
            x = layer(x, adj)
        return x

# 시간적 특징 추출 컨볼루션
class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # 1D 컨볼루션 -> batch norm -> ReLU
        return F.relu(self.norm(self.conv(x)))

# 특징 통합 후 차원 축소
class FeatureIntegrator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=64, stride=64):
        super(FeatureIntegrator, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # 입력 (batch_size, feature_dim, channels, length)
        batch_size, feature_dim, channels, length = x.size()

        # feature_dim 과 length 결합하여 새로운 차원 생성
        # (batch_size, channels, feature_dim * length)
        x = x.reshape(batch_size, channels, feature_dim * length)

        # 1D 컨볼루션
        x = self.conv(x)  # 출력 크기 (batch_size, out_channels, new_length)

        return x

# 슬리이딩 윈도우 방식으로 데이터 처리
class SlidingWindowProcessor(nn.Module):
    def __init__(self, model_dim, num_heads, window_size, stride):
        super(SlidingWindowProcessor, self).__init__()
        self.window_size = window_size                                  # 윈도우 사이즈
        self.stride = stride                                            # 슬라이딩 간격
        self.layer_norm1 = nn.LayerNorm([window_size, model_dim])
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.tcn_block = TemporalConvBlock(in_channels=model_dim, out_channels=32)
        self.fusion_conv = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        batch_size, _, length = x.shape
        window_outputs = []                 # 슬라이딩 윈도우의 출력 저장

        # 슬라이딩 윈도우 적용
        for window_start in range(0, length - self.window_size + 1, self.stride):
            window_end = window_start + self.window_size
            window = x[:, :, window_start:window_end] # B, C , W : 슬라이딩 윈도우로 데이터 추출
            window = window.permute(0, 2, 1) # B W : 윈도우 형태 변경

            window = self.layer_norm1(window)                                       # layer Norm 
            attn_output, _ = self.multi_head_attention(window, window, window)      # multi head attention
            attn_output = self.layer_norm2(attn_output + window)                    # skip connection + layerNorm
            tcn_input = attn_output.permute(0, 2, 1)                                # temporalconvBlock 입력 방식으로 전환
            tcn_output = self.tcn_block(tcn_input)                                  # 시간적 컨볼루션 처리

            window_outputs.append(tcn_output)                                       # 결과 저장

        # 모든 윈도우 결과를 스텍으로 결합
        stacked_outputs = torch.stack(window_outputs, dim=2)
        stacked_outputs = stacked_outputs.permute(0, 3, 1, 2).reshape(batch_size, 32, -1)
        # 윈도우 출력 병합
        fused_output = self.fusion_conv(stacked_outputs)

        return fused_output

