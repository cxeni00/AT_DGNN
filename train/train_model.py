import torch
from utils.utils import *
from config.config import *

CUDA = torch.cuda.is_available()
_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()


def train_one_epoch(data_loader, net, loss_fn, optimizer):                  
    """
    한 에폭 동안 모델을 훈련
    data_loader: 훈련 데이터 로더
    net: 훈련할 모델
    loss_fn: 손실 함수
    optimizer: 옵티마이저
    return: 평균 손실, 예측값 리스트, 실제값 리스트
    """
    net.train()                                                             # 모델을 학습 모드로 설정
    tl = Averager()                                                         # 손실 평균 계산을 위한 객체
    pred_train = []                                                         # 예측값 저장
    act_train = []                                                          # 실제값 저장
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()               # 데이터를 gpu로 이동

        out = net(x_batch)                                                  # 모델에 입력 데이터 전달                        
        loss = loss_fn(out, y_batch)                                        # 손실 계산
        _, pred = torch.max(out, 1)                                         # 예측값 추출
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()                                               # 이전 그래디언트 초기화
        loss.backward()                                                     # 그래디언트 계산
        optimizer.step()                                                    # 파라미터 업데이트
        tl.add(loss.item())                                                 # 손실 누적
    return tl.item(), pred_train, act_train                                 # 평균손실 / 예측값 리스트 / 실제값 리스트


def predict(data_loader, net, loss_fn): 
    """
    모델 예측 수행
    data_loader: 검증 또는 테스트 데이터 로더
    net: 예측할 모델
    loss_fn: 손실 함수
    return: 평균 손실, 예측값 리스트, 실제값 리스트
    """                                    
    net.eval()                                                              # 모델을 평가 모드로 설정
    pred_val = []                                                           # 예측값 저장 리스트
    act_val = []                                                            # 실제값 저장 리스트
    vl = Averager()                                                         # 손실 평균 계산을 위한 객체
    with torch.no_grad():                                                   # 그래디언트 비활성화
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())                                             # 손실 누적
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


def set_up(args):
    """
    초기 설정 수행 (GPU 설정, 경로 생성 등)
    args: 설정 인자를 포함한 객체
    """
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_train, label_train, data_val, label_val, subject, fold):
    """
    모델 훈련 및 검증
    args: 설정 인자
    data_train: 훈련 데이터
    label_train: 훈련 라벨
    data_val: 검증 데이터
    label_val: 검증 라벨
    subject: 피험자 ID
    fold: 교차 검증 fold 번호
    return: 최고 정확도와 F1 점수
    """
    seed_all(args.random_seed)                                                  # 랜덤 시드 설정                                                
    save_name = '_sub' + str(subject) + '_fold' + str(fold)                     # 저장 파일 이름 설정
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)     # 훈련 데이터 로더 생성
    val_loader = get_dataloader(data_val, label_val, args.batch_size)           # 검증 데이터 로더 생성

    model = get_model(args)                                                     # 모델 생성

    if CUDA:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)    # 옵티마이저 설정

    # 손실 함수 설정
    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)
    else:
        loss_fn = nn.CrossEntropyLoss()

    def save_model(name):
        """
        모델 저장
        name: 저장 파일 이름
        """
        previous_model = os.path.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)                                                           # 이전 모델 제거
        torch.save(model.state_dict(), os.path.join(args.save_path, '{}.pth'.format(name)))

    # 훈련 로그 초기화
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['F1'] = 0.0

    timer = Timer()
    patient = args.patient                          # early stoppin 기준 에폭 수
    counter = 0

    for epoch in range(1, args.max_epoch + 1):
        # 한 에폭 동안 훈련 수행
        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)

        acc_train, f1_train, _ = get_metrics(y_pred=pred_train, y_true=act_train)           # 성능 평가
        print('epoch {}, for the train set, loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss_train, acc_train, f1_train))
        
        # 검증 데이터로 예측 수행
        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn
        )
        acc_val, f1_val, _ = get_metrics(y_pred=pred_val, y_true=act_val)
        print('epoch {}, for the validation set, loss={:.4f} acc={:.4f} f1={:.4f}'.
              format(epoch, loss_val, acc_val, f1_val))
        
        # 검증 정확도가 최고 값이면 모델 저장
        if acc_val >= trlog['max_acc']:
            trlog['max_acc'] = acc_val
            trlog['F1'] = f1_val
            save_model('candidate')                 # 모델 저장
            counter = 0                             # 카운터 초기화
        else:
            counter += 1
            if counter >= patient:                  # early stopping 조건
                print('early stopping')
                break
        
        # 로그에 손실 및 정확도 기록
        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)

        print('ETA:{}/{} SUB:{} FOLD:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                subject, fold))
    # 훈련 로그 저장
    save_name = 'trlog' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = os.path.join(args.save_path, experiment_setting, 'log_train')
    ensure_path(save_path)
    torch.save(trlog, os.path.join(save_path, save_name))

    return trlog['max_acc'], trlog['F1']


def test(args, data, label, reproduce, subject, fold):
    """
    모델 테스트 수행
    args: 설정 인자
    data: 테스트 데이터
    label: 테스트 라벨
    reproduce: 모델 재현 여부
    subject: 피험자 ID
    fold: 교차 검증 fold 번호
    return: 테스트 정확도, 예측값, 실제값
    """
    set_up(args)                                                        # 기본 설정 수행
    seed_all(args.random_seed)                                          # 랜덤 시드 설정
    test_loader = get_dataloader(data, label, args.batch_size)          # 테스트 데이터 로더 생성

    model = get_model(args)                                             # 모델 생성
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()                                     # 손실 함수 설정

    # 특정 저장 경로에서 모델 로드
    if reproduce:
        model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
        data_type = 'model_{}_{}'.format(args.data_format, args.label_type)
        experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
        load_path_final = os.path.join(args.save_path, experiment_setting, data_type, model_name_reproduce)
        model.load_state_dict(torch.load(load_path_final))
    else:
        # 제공된 경로에서 모델 로드
        model.load_state_dict(torch.load(args.load_path_final))

    # 테스트 데이터로 예측 수행
    loss, pred, act = predict(
        data_loader=test_loader, net=model, loss_fn=loss_fn
    )
    acc, f1, cm = get_metrics(y_pred=pred, y_true=act)      # 성능 평가
    print('>>> Test:  loss={:.4f} acc={:.4f} f1={:.4f}'.format(loss, acc, f1))
    return acc, pred, act


def combine_train(args, data_train, label_train, subject, fold, target_acc):
    """
    두 번째 훈련 단계 수행 (Fine-Tuning)
    args: 설정 인자
    data_train: 훈련 데이터
    label_train: 훈련 라벨
    subject: 피험자 ID
    fold: 교차 검증 fold 번호
    target_acc: 조기 종료 기준 정확도
    """
    save_name = '_sub' + str(subject) + '_fold' + str(fold)
    set_up(args)                                                # 기본 설정 수행
    seed_all(args.random_seed)                                  # 랜덤 시드 설정

    train_loader = get_dataloader(data_train, label_train, args.batch_size)         # 훈련 데이터 로더 생성
    model = get_model(args)                                                         # 모델 생성
    if CUDA:
        model = model.cuda()
    model.load_state_dict(torch.load(args.load_path))                               # 이전에 저장한 모델 로드

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate * 1e-1)  # 옵티마이저 설정

    # 손실 함수 설정
    if args.LS:
        loss_fn = LabelSmoothing(args.LS_rate)                                      # 라벨 스무딩 사용시
    else:
        loss_fn = nn.CrossEntropyLoss()                                             # 기본 교차 엔트로피 손실 함수

    def save_model(name):
        previous_model = os.path.join(args.save_path, '{}.pth'.format(name))
        if os.path.exists(previous_model):
            os.remove(previous_model)
        torch.save(model.state_dict(), os.path.join(args.save_path, '{}.pth'.format(name)))

    # 훈련 로그 초기화
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['F1'] = 0.0

    timer = Timer()                     # 훈련 시간 측정

    for epoch in range(1, args.max_epoch_cmb + 1):
        # 한 에폭동안 훈련
        loss, pred, act = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer
        )
        acc, f1, _ = get_metrics(y_pred=pred, y_true=act)       # 성능 평가
        print('Stage 2 : epoch {}, for train set loss={:.4f} acc={:.4f} f1={:.4f}'
              .format(epoch, loss, acc, f1))

        # 조기 종료 조건
        if acc >= target_acc or epoch == args.max_epoch_cmb:
            print('early stopping!')
            save_model('final_model')                                                       # 최종 모델 저장
            # 모델 재현을 위한 저장
            model_name_reproduce = 'sub' + str(subject) + '_fold' + str(fold) + '.pth'
            data_type = 'model_{}_{}'.format(args.data_format, args.label_type)
            experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
            save_path = os.path.join(args.save_path, experiment_setting, data_type)
            ensure_path(save_path)
            model_name_reproduce = os.path.join(save_path, model_name_reproduce)
            torch.save(model.state_dict(), model_name_reproduce)
            break

        # 훈련 로그 갱신
        trlog['train_loss'].append(loss)
        trlog['train_acc'].append(acc)

        print('ETA:{}/{} SUB:{} TRIAL:{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch),
                                                 subject, fold))

    # 훈련 로그 저장
    save_name = 'trlog_comb' + save_name
    experiment_setting = 'T_{}_pool_{}'.format(args.T, args.pool)
    save_path = os.path.join(args.save_path, experiment_setting, 'log_train_cmb')
    ensure_path(save_path)
    torch.save(trlog, os.path.join(save_path, save_name))
