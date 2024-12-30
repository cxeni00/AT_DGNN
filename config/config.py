import argparse
import sys

def set_config():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data-path', type=str, default='/Users/cxeni/my_AT_DGNN/AT-DGNN/data', help = "데이터 경로")
    parser.add_argument('--subjects', type=int, default=32, help = "총 피험자 수")
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4] , help = "분류할 클래스 수")
    parser.add_argument('--label-type', type=str, default='A', choices=['A', 'V', 'D', 'L'] , help = "라벨 유형 A:Arousal, V:Valence, D:Dominance, L:Liking")
    parser.add_argument('--segment', type=int, default=4 , help = "세그먼트 길이 초 단위") 
    parser.add_argument('--overlap', type=float, default=0 , help = "세그먼트 간 겹치는 비율")
    parser.add_argument('--sampling-rate', type=int, default=1000 , help = "원본 데이터 sampling rate")
    parser.add_argument('--target-rate', type=int, default=200 , help = "목표 sampling rate")
    parser.add_argument('--trial-duration', type=int, default=59, help='실험 지속 시간(초)')
    parser.add_argument('--input-shape', type=str, default="1,32,800" , help = "입력 데이터의 모양(채널, 샘플크기)") 
    parser.add_argument('--data-format', type=str, default='eeg' , help = "데이터 형식")
    parser.add_argument('--bandpass', type=tuple, default=(1, 50) , help = "필터링 대역폭(1,50)")
    parser.add_argument('--channels', type=int, default=32 , help = "사용할 채널 수")

    # Training Process
    parser.add_argument('--fold', type=int, default=10 , help = "교차 검증 fold 수")
    parser.add_argument('--random-seed', type=int, default=3407 , help = "랜덤 시드")
    parser.add_argument('--max-epoch', type=int, default=200 , help = "최대 에폭 수")
    parser.add_argument('--patient', type=int, default=40 , help = "early stopping 에폭 수")  # 20
    parser.add_argument('--patient-cmb', type=int, default=10 , help = "combine early stopping 에폭 수")  # 8
    parser.add_argument('--max-epoch-cmb', type=int, default=40 , help = "combine 최대 에폭 수")  # 20
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)  #1e-3
    parser.add_argument('--training-rate', type=float, default=0.8)
    parser.add_argument('--weight-decay', type=float, default=0.001 , help = "L2 정규화를 위한 가중치 감쇠") 
    parser.add_argument('--step-size', type=int, default=5 , help = "학습률 스케쥴링 단계 크기")
    parser.add_argument('--dropout', type=float, default=0.5) 
    parser.add_argument('--LS', type=bool, default=True, help="Label smoothing 활성화 여부") 
    parser.add_argument('--LS-rate', type=float, default=0.1 , help = "Label smoothing 비율")
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--balance', type=bool, default=False , help = "데이터 균형 여부")

    # saving and loading
    parser.add_argument('--save-path', default='./save/' , help = "모델 및 결과 저장 경로")
    parser.add_argument('--load-path', default='./save/max-acc.pth' , help = "최대 정확도 모델")
    parser.add_argument('--load-path-final', default='./save/final_model.pth' , help = "최종 모델")
    parser.add_argument('--save-model', type=bool, default=True , help = "모델 저장 여부")

    # Model Parameters
    parser.add_argument('--pool', type=int, default=16 , help = "폴링크기")
    parser.add_argument('--pool-step-rate', type=float, default=0.25 , help = "폴링 이동 비율")
    parser.add_argument('--T', type=int, default=64 , help = "시간적 필터 수")
    parser.add_argument('--graph-type', type=str, default='fro', choices=['fro', 'gen', 'hem', 'BL'] , help = "그래프 유형")
    parser.add_argument('--hidden', type=int, default=32 , help = "히든 레이터 크기") 

    # Reproduce the result using the saved model
    parser.add_argument('--reproduce', action='store_true', default=False)

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args([])  # 빈 리스트로 파싱
    else:
        args = parser.parse_args()
    gpu = args.gpu
    
    # Convert the input shape from string to tuple of integers
    args.input_shape = tuple(map(int, args.input_shape.split(',')))

    return args, gpu
