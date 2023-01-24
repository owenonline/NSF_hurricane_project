from random import shuffle
import torch
from data_loader_multi_task2 import DisasterDataset
from torch.utils.data import DataLoader
from torchvision.models import resnet
from torch import nn
from tqdm import tqdm
from gsam import GSAM, LinearScheduler
# from model.smooth_cross_entropy import smooth_crossentropy
import torch.nn.functional as F

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)

    return F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
#import resnet_example

# ===========================
epochs = 100

tasks = ['disaster_types','damage_severity','humanitarian','informative']
# tasks = ['disaster_types']
for task in tasks:

   train_set = DisasterDataset(task = task,split='train')
   train_loader = DataLoader(train_set, batch_size=64,pin_memory=True,shuffle=True, num_workers=8)


   test_set = DisasterDataset(task = task,split='test')
   test_loader = DataLoader(test_set, batch_size=4,pin_memory=True, num_workers=8)

   dev_set = DisasterDataset(task = task,split='dev')
   dev_loader = DataLoader(dev_set, batch_size=4,pin_memory=True, num_workers=8)

   # import pdb; pdb.set_trace()
   res_mod = resnet.resnet18(pretrained=True)
   num_ftrs = res_mod.fc.in_features
   res_mod.fc = nn.Linear(num_ftrs, len(train_set.label_2_id[task]))
   model = res_mod

   model = model.cuda()
   model = nn.DataParallel(model)
   # ===========================
   class_weights = torch.FloatTensor(train_set.class_weights).cuda()

   criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

   # optimizer = torch.optim.AdamW (model.parameters(),lr=1e-3, weight_decay=1e-1)
   optimizer = torch.optim.AdamW (model.parameters(),lr=1e-5, weight_decay=1e-2)

   # rho_max, rho_min, alpha, label_smoothing = 2.0, 2.0, 0.2, 0.1
   rho_max, rho_min, alpha, label_smoothing = 2.0, 2.0, 0.2, 0.1
   lr_scheduler = LinearScheduler(T_max=epochs*len(train_loader), max_value=1e-5, min_value=1e-5*0.01, optimizer=optimizer)
   rho_scheduler = LinearScheduler(T_max=epochs*len(train_loader), max_value=rho_max, min_value=rho_min)
   gsam_optimizer = GSAM(params=model.parameters(), base_optimizer=optimizer, model=model, gsam_alpha=alpha, rho_scheduler=rho_scheduler, adaptive=True)


   # lr_scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience=10)
   # import pdb; pdb.set_trace()
   
   



   best_val_acc, best_test_acc =  .0, .0
   for ep in range(epochs):
      print(ep)
      correct = 0
      model.train()     
      for id, batch in tqdm(enumerate(train_loader)):
         img, label = batch
         label = label.long()
         img, label = img.cuda(), label.cuda()


         def loss_fn(predictions, targets):
            return smooth_crossentropy(predictions, targets, smoothing=label_smoothing).mean()

         gsam_optimizer.set_closure(loss_fn, img, label)

         preds, loss = gsam_optimizer.step()
    



         # optimizer.zero_grad()


         # with torch.enable_grad():
         #    logits = model(img)
         #    loss = criterion(logits, label)
         # preds = model(img)
         with torch.no_grad():
            correct += (torch.argmax(preds,1)==label).sum()
            # log(model, loss.cpu().repeat(args.batch_size), correct.cpu(), scheduler.lr())
            lr_scheduler.step()
            gsam_optimizer.update_rho_t()

         # loss.backward()
         # optimizer.step()
         # lr_scheduler.step()
      train_acc = correct/len(train_set)
      model.eval()  
      with torch.no_grad():    
         correct = 0
         for id, batch in tqdm(enumerate(dev_loader)):
            img, label = batch
            label = label.long()
            img, label = img.cuda(), label.cuda()

            preds = model(img)
            correct += (torch.argmax(preds,1)==label).sum()
         val_acc = correct/len(dev_set)
         if val_acc > best_val_acc:
            correct = 0
            for id, batch in tqdm(enumerate(test_loader)):
               img, label = batch
               label = label.long()
               img, label = img.cuda(), label.cuda()

               preds = model(img)
               correct += (torch.argmax(preds,1)==label).sum()
            best_test_acc = correct/len(test_set)
         with open('output_balanced_gsam.txt','a') as f:
            curr_lr = [group['lr'] for group in lr_scheduler.optimizer.param_groups][0]
            f.write(task +'_epoch: '+str(ep) + '_test:'+str(best_test_acc) + '_val:'+str(val_acc)+ '_train:'+str(train_acc)+'_lr:'+str(curr_lr)+'\n')
         # lr_scheduler.step(val_acc)
         

