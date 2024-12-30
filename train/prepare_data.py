import pickle as cPickle
from scipy import signal
from train.train_model import *
from scipy.signal import resample
from torch.utils.data import Dataset
import os
import numpy as np
import h5py

# EEG 데이터셋 클래스
class eegDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor                           # 입력 데이터        
        self.y = y_tensor                           # 라벨 데이터

        assert self.x.size(0) == self.y.size(0)     # 입력 데이터 = 샘플 개수

    # 주어진 인덱스에 해당하는 데이터와 라벨 반환
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # 데이터셋의 샘플 수 반환
    def __len__(self):
        return len(self.y)

# EEG 데이터 전처리 및 저장
class PrepareData:
    def __init__(self, args):
    
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path            
        self.label_type = args.label_type
        
        # EEG 채널의 초기 순서
        self.original_order = ['Fp1', 'Fp2', 'AF3', 'AF4', 'Fz', 'F3', 'F4', 'F7', 'F8',
                                'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8',
                                'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8',
                                'PO3', 'PO4', 'Oz', 'O1', 'O2']
        
        # 그래프 설계
        self.graph_fro_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_gen_DEAP = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                               ['FC5', 'FC1', 'FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_hem_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz', 'Cz', 'Pz', 'Oz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3'], ['C4'], ['CP5', 'CP1'], ['CP2', 'CP6'],
                               ['P7', 'P3'], ['P4', 'P8'], ['PO3', 'O1'], ['PO4', 'O2'], ['T7'], ['T8']]
        

        self.graph_type = args.graph_type

    def run(self, subject_list, split, expand):
        """
        주어진 피험자 데이터를 처리하고 저장

        Parameters
        ----------
        subject_list: subject_list: 처리할 피험자 목록
        split: 데이터를 세그먼트로 분할 여부
        expand: CNN 사용을 위한 차원 확장 여부

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        for sub in subject_list:
            # 피험자 데이터와 라벨 로드
            data_, label_ = self.load_data_per_subject(sub)

            # 라벨 유형 선택
            label_ = self.label_selection(label_)

            # 데이터 전처리
            data_, label_ = self.preprocess_data(data=data_, label=label_, split=split, expand=expand)

            print('Data and label prepared!')
            print('sample_' + str(sub + 1) + '.dat')
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')

            # 전처리된 데이터 저장
            self.save(data_, label_, sub)

        self.args.sampling_rate = self.args.target_rate

    def load_data_per_subject(self, sub):
        """
        특정 피험자의 원본 데이터를 로드

        Parameters
        ----------
        sub: 로드할 피험자 ID

        Returns
        -------
        data: (40, 32, 7680) label: (40, 4)
        """
        sub += 1
        sub_code = str('sample_' + str(sub) + '.dat')

        # 데이터 경로 구성 및 로드
        subject_path = os.path.join(self.data_path, sub_code)
        subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
        label = subject['labels']
        data = subject['data']

        # EEG 채널 재 정렬
        data = self.reorder_channel(data=data, graph=self.graph_type)
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        return data, label

    def reorder_channel(self, data, graph):
        """
        EEG 채널을 그래프 설계에 따라 재정렬

        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'fro':
            graph_idx = self.graph_fro_DEAP
        elif graph == 'gen':
            graph_idx = self.graph_gen_DEAP
        elif graph == 'hem':
            graph_idx = self.graph_hem_DEAP
        elif graph == 'BL':
            graph_idx = self.original_order

        idx = []
        if graph in ['BL']:
            for chan in graph_idx:
                idx.append(self.original_order.index(chan))
        else:
            num_chan_local_graph = []
            for i in range(len(graph_idx)):
                num_chan_local_graph.append(len(graph_idx[i]))
                for chan in graph_idx[i]:
                    idx.append(self.original_order.index(chan))

            # 로컬 그래프의 채널 수 저장
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
        return data[:, idx, :]

    def label_selection(self, label):
        """
        1. 사용할 라벨 차원 선택
        2. 이진 라벨 선택

        Parameters
        ----------
        label: (trial, 4)

        Returns
        -------
        label: (trial,)
        """
        if self.label_type == 'V':              # valence
            label = label[:, 0]
        elif self.label_type == 'A':            # arousal
            label = label[:, 1]
        elif self.label_type == 'D':            # diminance
            label = label[:, 2]
        elif self.label_type == 'L':            # liking
            label = label[:, 3]
        return label

    def save(self, data, label, sub):
        """
        전처리된 데이터를 저장된 경로에 저장

        Parameters
        ----------
        data: 처리된 데이터
        label: 해당 라벨
        sub: subject ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, 'MEEG', self.args.label_type)
        save_path = os.path.join(save_path, data_type)

        # 경로가 없으면 생성
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        
        # 데이터 저장
        name = 'sub' + str(sub) + '.hdf'
        save_path = os.path.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    def preprocess_data(self, data, label, split, expand):
        """
        EEG 데이터 전처리

        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        split: 세그먼트로 나눌지 여부
        expand: CNN용 추가 차원 생성 여부

        Returns
        -------
        preprocessed
        data: (trial, channel, target_length)
        label: (trial,)
        """
        if expand:
            # 추가 차우너을 생성해 CNN 입력에 맞춤
            data = np.expand_dims(data, axis=-3)

        # 대역 통과 필터 적용
        data = self.bandpass_filter(data=data, lowcut=self.args.bandpass[0], highcut=self.args.bandpass[1],
                                    fs=self.args.sampling_rate, order=5)
        
        # 노치 필터 (50Hz)
        data = self.notch_filter(data=data, fs=self.args.sampling_rate, Q=50)

        # 다운 샘플링
        if self.args.sampling_rate != self.args.target_rate:
            data, label = self.downsample_data(
                data=data, label=label, sampling_rate=self.args.sampling_rate,
                target_rate=self.args.target_rate)

        # 데이터를 세그먼트로 분할
        if split:
            data, label = self.split(data, label, self.args.segment, self.args.overlap, self.args.target_rate)

        return data, label

    def split(self, data, label, segment_length, overlap, sampling_rate):
        """
        데이터를 세그먼트로 분할

        Parameters
        ----------
        data: (trial, f, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, f, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))          # 세그먼트 간 이동 크기
        data_segment = sampling_rate * segment_length                       # 세그먼트 크기
        data_split = []

        # 세그먼트 개수 계산
        number_segment = int((data_shape[-1] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)                                 # 분할된 데이터를 새로운 차원으로 스텍
        label = np.stack([np.repeat(label[i], int(number_segment + 1)) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)                                       # 데이터와 라벨 개수 동일
       
        return data, label

    def extract_features(self, data, sfreq):
        """
        EEG 데이터에서 주파수 대역별 특징 추출
        data : (20, 14, 1, 32, 800)
        sfreq : 샘플링 레이트

        returns : 대역별 특징 추출 데이터(20,14,1,32,5)
        """
        num_trials, num_segments, _, num_channels, num_samples = data.shape
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 14),
            'beta': (14, 31),
            'gamma': (31, 50)
        }
        features = np.zeros((num_trials, num_segments, 1, num_channels, len(bands)))

        for i, (band_name, band_range) in enumerate(bands.items()):
            for trial in range(num_trials):
                for segment in range(num_segments):
                    for channel in range(num_channels):
                        channel_data = data[trial, segment, 0, channel, :]
                        # 단일 채널 테이터 필터링 후 특징 계산
                        filtered_data = self.bandpass_filter(channel_data, band_range[0], band_range[1], sfreq)
                        features[trial, segment, 0, channel, i] = np.log(np.var(filtered_data) + 1e-8)

        return features

    def downsample_data(self, data, label, sampling_rate, target_rate):
        """
        데이터를 목표 샘플링 레이트로 다운 샘플링

        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        sampling_rate: original sampling rate
        target_rate: target sampling rate

        Returns
        -------
        downsampled data: (trial, channel, target_length)
        label: (trial,)
        """
        target_length = int(data.shape[-1] * target_rate / sampling_rate)
        downsampled_data = resample(data, target_length, axis=-1)
        return downsampled_data, label

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        대역 통과 필터를 적용

        Parameters
        ----------
        data: (trial, channel, data)
        lowcut: low cut frequency
        highcut: high cut frequency
        fs: sampling rate
        order: filter order

        Returns
        -------
        filtered data: (trial, channel, data)
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='bandpass')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        return filtered_data

    def notch_filter(self, data, fs, Q=50):
        """
        노치 필터 50Hz

        Parameters
        ----------
        data: (trial, channel, data)
        fs: sampling rate
        Q: 필터 품질 계수

        Returns
        -------
        filtered data: (trial, channel, data)
        """
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.notch_filter_per_channel(data[i, j, :], fs, Q)
        return data

    def notch_filter_per_channel(self, param, fs, Q):
        """
        단일 채널에 노치 필터 적용
        
        Parameters
        ----------
        param: (data,)
        fs: sampling rate
        Q: Q value for notch filter

        Returns
        -------
        filtered data: (data,)
        """
        w0 = Q / fs
        b, a = signal.iirnotch(w0, Q)
        param = signal.filtfilt(b, a, param)
        return param
