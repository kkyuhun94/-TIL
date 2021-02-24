import torch
from tqdm.auto import tqdm
from utils import adjust_learning_rate


def train(model, criterion, optimizer, loaders, args) :
    trn_loss_list = []
    val_loss_list = []

    trainLoader, valLoader, testLoader = loaders

    num_epochs = args.num_epoch 
    num_batches = len(trainLoader)
    device = args.device
    
    for epoch in tqdm(range(num_epochs)):
        adjust_learning_rate(optimizer, epoch, args.learning_rate) # utils.py
        trn_loss = 0.0
        for i, data in enumerate(trainLoader):
            x, label = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            model_output = model(x)
            loss = criterion(model_output, label)
            loss.backward()
            optimizer.step()
            
            trn_loss += loss.item()
            
            if (i+1) % (num_batches//2) == 0: 
                with torch.no_grad(): 
                    val_loss = 0.0
                    corr_num = 0
                    total_num = 0
                    for j, val in enumerate(valLoader):
                        val_x, val_label = val[0].to(device), val[1].to(device)
                        val_output = model(val_x)
                        v_loss = criterion(val_output, val_label)
                        val_loss += v_loss
                        
                        model_label = val_output.argmax(dim=1)
                        corr = torch.eq(val_label, model_label).sum()
                        corr_num += corr.item()
                        total_num += val_label.size(0)
                
                print(f"epoch: {epoch+1:03d}/{args.num_epochs} | "
                      f"step: {i+1:03d}/{num_batches} | "
                      f"trn loss: {trn_loss/100:08.4f} "
                      f"| val loss: {val_loss/len(valLoader):08.4f} "
                      f"| acc: {(corr_num/total_num)*100:.2f}")        
                
                trn_loss_list.append(trn_loss/100)
                val_loss_list.append(val_loss/len(valLoader))
                trn_loss = 0.0

    print("training finished!")
    test(model,testLoader,args)
    




def test(model,testloader,args) :

    device = args.device
    # test acc
    # resnet 152, start 0.01 , bs 64 , optimizer SGD momentum0.9 weight decay 0.0002
    with torch.no_grad():
        corr_num = 0
        total_num = 0
        for j, val in enumerate(testloader):
            val_x, val_label = val
            val_x = val_x.to(device)
            val_label =val_label.to(device)
            val_output = model(val_x)
            model_label = val_output.argmax(dim=1)
            corr = val_label[val_label == model_label].size(0)
            corr_num += corr
            total_num += val_label.size(0)

    print("test_acc: {:.2f}".format(corr_num / total_num * 100))