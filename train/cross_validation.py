import copy
import datetime

from config.config import *
from sklearn.model_selection import KFold
from train.train_model import *
from utils.utils import *

# 현재 작업 디랙토리 저장
ROOT = os.getcwd()

# GPU 설정 초기화 및 환경 변수 설정
_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()

# CrossValidation 클래스 : 데이터셋에서 교차 검증을 수행하는 데 사용
class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None

        # 결과를 저장할 경로를 생성하고 로그 파일을 초기화
        result_path = os.path.join(args.save_path, 'result')
        ensure_path(result_path) # 폴더가 없으면 생성

        # 로그 파일에 실험 환경 설정 기록
        self.text_file = os.path.join(result_path, "results_{}.txt".format('MEEG'))
        file = open(self.text_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str('AT-DGNN') + ' on ' + str('MEEG') +
                   "\n1)number_class:" + str(args.num_class) +
                   "\n2)random_seed:" + str(args.random_seed) +
                   "\n3)learning_rate:" + str(args.learning_rate) +
                   "\n4)training_rate:" + str(args.training_rate) +
                   "\n5)pool:" + str(args.pool) +
                   "\n6)num_epochs:" + str(args.max_epoch) +
                   "\n7)batch_size:" + str(args.batch_size) +
                   "\n8)dropout:" + str(args.dropout) +
                   "\n9)hidden_node:" + str(args.hidden) +
                   "\n10)input_shape:" + str(args.input_shape) +
                   "\n11)class:" + str(args.label_type) +
                   "\n12)T:" + str(args.T) +
                   "\n13)graph-type:" + str(args.graph_type) +
                   "\n14)patient:" + str(args.patient) +
                   "\n15)patient-cmb:" + str(args.patient_cmb) +
                   "\n16)max-epoch-cmb:" + str(args.max_epoch_cmb) +
                   "\n17)fold:" + str(args.fold) +
                   "\n18)model:" + str('AT-DGNN') +
                   "\n19)data-path:" + str(args.data_path) +
                   "\n20)balance:" + str(args.balance) +
                   "\n21)bandpass:" + str(args.bandpass) +
                   "\n22)dataset:" + str('MEEG') +
                    "\n23)overlap:" + str(args.overlap) +
                   '\n')
        file.close()

    def load_per_subject(self, sub):
        """
        load data for sub
        param sub: which subject's data to load
        return: data and label
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, 'MEEG', self.args.label_type)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = os.path.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label

    def prepare_data(self, idx_train, idx_test, data, label):
        """
        1. get training and testing data according to the index
        2. numpy.array-->torch.tensor
        param idx_train: index of training data
        param idx_test: index of testing data
        param data: (segments, 1, channel, data)
        param label: (segments,)
        return: data and label
        """
        # 훈련 데이터와 테스트 데이터 분리
        data_train = data[idx_train]
        label_train = label[idx_train]
        data_test = data[idx_test]
        label_test = label[idx_test]

        # 훈련 데이터와 테스트 데이터를 출소하여 텐서 형태로 변환
        data_train = np.concatenate(data_train, axis=0)
        label_train = np.concatenate(label_train, axis=0)
        if len(data_test.shape) > 4:
            data_test = np.concatenate(data_test, axis=0)
            label_test = np.concatenate(label_test, axis=0)

        # 데이터 정규화
        data_train, data_test = self.normalize(train=data_train, test=data_test)

        # PyTorch 텐서 변환
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()
        return data_train, label_train, data_test, label_test

    def normalize(self, train, test):
        """
        this function do standard normalization for EEG channel by channel
        :param train: training data (sample, 1, chan, datapoint)
        :param test: testing data (sample, 1, chan, datapoint)
        :return: normalized training and testing data
        """
        # data: sample x 1 x channel x data
        for channel in range(train.shape[2]):
            mean = np.mean(train[:, :, channel, :])
            std = np.std(train[:, :, channel, :])
            train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
            test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        """
        Get the validation set using the same percentage of the two classe samples
        param data: training data (segment, 1, channel, data)
        param label: (segments,)
        param train_rate: the percentage of trianing data
        param random: bool, whether to shuffle the training data before get the validation data
        return: data_trian, label_train, and data_val, label_val
        """
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments
        # 라벨이 0인 데이터와 1인 데이터 분리 
        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class 1
        index_random_1 = copy.deepcopy(index_1)

        # 랜덤하게 데이터 섞기
        if random:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)

        # 훈련 및 검증 데이터 인덱스 생성
        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)],
                                      index_random_1[:int(len(index_random_1) * train_rate)]),
                                     axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):],
                                    index_random_1[int(len(index_random_1) * train_rate):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label

    def n_fold_CV(self, subject, fold, reproduce):
        """
        this function achieves n-fold cross-validation
        param subject: how many subjects to load
        param fold: how many fold.
        """
        # 각 피험자 별로 학습 및 검증 수행
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1

        for sub in subject:
            # 필험자 데이터 로드
            data, label = self.load_per_subject(sub)
            va_val = Averager() # 검증 정확도와 평균 계산 
            vf_val = Averager() # 검증 F1 점수와 평균 계산
            preds, acts = [], [] # 예측값, 실제값

            # 외부 kFold 교차 검증 시작
            kf = KFold(n_splits=fold, shuffle=True)
            for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data)):
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))
                # 데이터 준비
                data_train, label_train, data_test, label_test = self.prepare_data(
                    idx_train=idx_train, idx_test=idx_test, data=data, label=label)
                # 데이터 밸런스 조정 여부 확인
                if self.args.balance:
                    data_train, label_train, data_val, label_val = self.split_balance_class(
                        data=data_train, label=label_train, train_rate=self.args.training_rate, random=True)
                    
                # 재현 모드인지 새로운 모델 학습하는지 확인
                if reproduce:
                    # to reproduce the reported ACC
                    acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=reproduce,
                                               subject=sub, fold=idx_fold)
                    acc_val = 0
                    f1_val = 0    # 검증 결과는 재현 모드에서는 계산하지 않음
                else:
                    # 새로운 모델 학습
                    print('Training:', data_train.size(), label_train.size())
                    print('Test:', data_test.size(), label_test.size())
                    acc_val, f1_val = self.first_stage(data=data_train, label=label_train,
                                                       subject=sub, fold=idx_fold)
                    # 추라 학습
                    combine_train(args=self.args,
                                  data_train=data_train, label_train=label_train,
                                  subject=sub, fold=idx_fold, target_acc=1)

                    # 테스트 점수와 f1 점수 계산
                    acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
                                               reproduce=reproduce,
                                               subject=sub, fold=idx_fold)
                # 결과 저장
                va_val.add(acc_val)
                vf_val.add(f1_val)
                preds.extend(pred)
                acts.extend(act)

            # 결과 fold 별로 저장
            tva.append(va_val.item())
            tvf.append(vf_val.item())
            acc, f1, _ = get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc)
            ttf.append(f1)

            # sub 별 최종 결과 출력
            result = 'sub {}: total test accuracy {}, f1: {}'.format(sub, tta[-1], f1)
            self.log2txt(result)

        # 결과 출력 및 로그
        tta = np.array(tta)
        ttf = np.array(ttf)
        tva = np.array(tva)
        tvf = np.array(tvf)
        mACC = np.mean(tta)
        mF1 = np.mean(ttf)
        std = np.std(tta)
        stdF1 = np.std(ttf)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)

        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: test mean F1:{}'.format(mF1))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: val mean F1:{}'.format(mF1_val))
        results = ('test mAcc={} std:{} mF1={} std:{} \n'
                   'val mAcc={} F1={}').format(mACC, std, mF1, stdF1, mACC_val, mF1_val)
        self.log2txt(results)

    def first_stage(self, data, label, subject, fold):
        """
        this function achieves n-fold-CV to:
        1. select hyper-parameters on training data
        2. get the model for evaluation on testing data
        param data: (segments, 1, channel, data)
        param label: (segments,)
        param subject: which subject the data belongs to
        param fold: which fold the data belongs to
        return: mean validation accuracy
        """
        # use n-fold-CV to select hyper-parameters on training data
        # save the best performance model and the corresponding acc for the second stage
        # data: trial x 1 x channel x time
        kf = KFold(n_splits=3, shuffle=True)
        va = Averager() # 정확도 평균 계산
        vf = Averager() # f1 평균 계산
        va_item = []
        maxAcc = 0.0
        for i, (idx_train, idx_val) in enumerate(kf.split(data)):
            print('Inner 3-fold-CV Fold:{}'.format(i))
            # 데이터 나누기
            data_train, label_train = data[idx_train], label[idx_train]
            data_val, label_val = data[idx_val], label[idx_val]
            # 훈련 및 검증
            acc_val, F1_val = train(args=self.args,
                                    data_train=data_train,
                                    label_train=label_train,
                                    data_val=data_val,
                                    label_val=label_val,
                                    subject=subject,
                                    fold=fold)
            va.add(acc_val)
            vf.add(F1_val)
            va_item.append(acc_val)
            # 최대 정확도 모델 저장
            if acc_val >= maxAcc:
                maxAcc = acc_val
                # choose the model with higher val acc as the model to second stage
                old_name = os.path.join(self.args.save_path, 'candidate.pth')
                new_name = os.path.join(self.args.save_path, 'max-acc.pth')
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
                print('New max ACC model saved, with the val ACC being:{}'.format(acc_val))

        mAcc = va.item()
        mF1 = vf.item()
        return mAcc, mF1

    def log2txt(self, content):
        """
        This function log the content to results.txt
        param content: string, the content to log.
        """
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()
