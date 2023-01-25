import torch
import sys
from data_loader import DisasterDataset
from torch.utils.data import DataLoader
from torchvision.models import resnet, mobilenet
from torch import nn
#import resnet_example

epochs = 100

# ===========================
if sys.argv[1] == "resnet":
   res_mod = resnet.resnet18(weights=resnet.ResNet18_Weights.DEFAULT)
   num_ftrs = res_mod.fc.in_features
   res_mod.fc = nn.Linear(num_ftrs, 7)
   model = res_mod
elif sys.argv[1] == "mobilenet":
   res_mod = mobilenet.mobilenet_v3_small(weights=mobilenet.MobileNet_V3_Small_Weights.DEFAULT)
   model = res_mod
elif sys.argv[1] == "resnext":
   res_mod = resnet.resnext101_64x4d(weights=resnet.ResNeXt101_64X4D_Weights.DEFAULT)
   num_ftrs = res_mod.fc.in_features
   res_mod.fc = nn.Linear(num_ftrs, 7)
   model = res_mod

model = model.cuda()
model = nn.DataParallel(model)
# ===========================

criterion = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(model.parameters(),lr=1e-5, weight_decay=1e-3)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-2)
lr_scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience=10)


train_set = DisasterDataset(split='train')
train_loader = DataLoader(train_set, batch_size=128, num_workers = 8, pin_memory=True)

test_set = DisasterDataset(split='test')
test_loader = DataLoader(test_set, batch_size=4, num_workers = 8, pin_memory=True)

dev_set = DisasterDataset(split='dev')
dev_loader = DataLoader(dev_set, batch_size=4, num_workers = 8, pin_memory=True)

best_val_acc, best_test_acc = 0.0, 0.0

# import pdb; pdb.set_trace()
for ep in range(epochs):
   print(ep)
   correct = 0

   for id, batch in enumerate(train_loader):
      img, label = batch
      label = label.long()
      img, label = img.cuda(), label.cuda()
      optimizer.zero_grad()

      with torch.enable_grad():
         logits = model(img)
         loss = criterion(logits, label)
      preds = model(img)
      correct += (torch.argmax(preds,1)==label).sum()
      loss.backward()
      optimizer.step()
   train_acc = correct/len(train_set)
   correct = 0
   for id, batch in enumerate(dev_loader):
      img, label = batch
      label = label.long()
      img, label = img.cuda(), label.cuda()

      preds = model(img)
      correct += (torch.argmax(preds,1)==label).sum()
   val_acc = correct/len(dev_set)
   if val_acc > best_val_acc:
      correct = 0
      for id, batch in enumerate(test_loader):
         img, label = batch
         label = label.long()
         img, label = img.cuda(), label.cuda()

         preds = model(img)
         correct += (torch.argmax(preds,1)==label).sum()
      best_test_acc = correct/len(test_set)
   with open('out/output_single_{}.txt'.format(sys.argv[1]),'a') as f:
      curr_lr = [group['lr'] for group in lr_scheduler.optimizer.param_groups][0]
      f.write('test:'+str(best_test_acc) + '_val:'+str(val_acc)+ '_train:'+str(train_acc)+'_lr:'+str(curr_lr)+'\n')
   lr_scheduler.step(val_acc)




