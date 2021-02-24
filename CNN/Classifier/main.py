import os
import numpy as np
# from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F 



import argparse
from argparse import Namespace # 이해 필요 

from dataloader import dataloader
from model import Resnet
from train_model import train



def main(args) :
    loaders = dataloader(args)
    model = Resnet(args.n_classes, args.modelname).to(args.device) 
    model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD( model.parameters(), momentum = args.momentum, weight_decay = args.weight_decay )
    train(model, criterion, optimizer, loaders, args)
    torch.save(model.state_dict(), 'model.pt')



# 해당 모듈이 import 된 경우가 아니라 인터프리터에서 직접 실행될 경우에만 if문 이하의 코드를 실행
# __name__ : interpreter가 실행전에 만들어 둔 글로벌 변수 

if __name__ == '__main__' : 
    # Argument Parsing
    # 인스턴스 생성
    parser = argparse.ArgumentParser()
    # 인자 추가 
    # parser.add_argument('--train', action = "store_true") # "store_true" : 인자가 입력되면 True 아니면 False로 인식 
    parser.add_argument('--seed', type = int, default = 777)
    parser.add_argument('--cuda', action = "store_ture" ) 
    parser.add_argument('--DATA_PATH', type = str, default ='data/Images')
    parser.add_argument('--test_pct', type = float , default = 0.2)
    parser.add_argument('--val_pct', type = float , default = 0.1)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--n_classes', type = int , default = 120)
    parser.add_argument('--learning_rate', type = float , default = 0.025)
    parser.add_argument('--momentum', type = float, default = 0.9)
    parser.add_argument('--weight_decay', type = float , default = 1e-5)
    parser.add_argument('--num_epoch', type = int , default = 200)
    parser.add_argument('--modelname', type = str , default = 'resnet152')

    args = parser.parse_args() # 객체로 변환, Namespace의 속성으로 설정
    torch.manual_seed(args.seed)

    if args.cuda and torch.cuda.is_available() :
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else :
        args.device = torch.device('cpu') 
    
    main(args)