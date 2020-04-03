from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import visdom
import argparse


def train_model(model, criterion, optimizer, scheduler, num_epochs,
                dataloader, device, dataset_sizes, output_path):
    # model = init_model(model)
    model.to(device)
    vis = visdom.Visdom()
    #     vis.close()
    loss_curve = {'train': [], 'val': []}
    acc = {'train': [], 'val': []}

    plot_loss_win = vis.line([np.zeros(2), np.zeros(2)])
    plot_acc = vis.line(np.zeros(2))

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        # show acc every x epochs
        if (epoch % 1 == 0) or (epoch == num_epochs):
            print('\tEpoch {}/{}'.format(epoch, num_epochs - 1))
            print('\t' + '-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels, headers in dataloader[phase]:
                inputs = inputs.to(device, dtype=torch.float)
                labels = torch.from_numpy(np.asarray(labels, dtype='int64'))
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phases
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics:
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            loss_curve[phase].append(epoch_loss)
            acc[phase].append(epoch_acc.cpu().numpy())

            if (epoch % 1 == 0) or (epoch == num_epochs):
                print('\t{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copu the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        X = np.linspace(0, epoch, epoch + 1)
        #         a = np.column_stack((X, X)),
        #         b = np.column_stack((loss_curve['train'], loss_curve['val']))
        vis.line(X=np.column_stack((X, X)),
                 Y=np.column_stack((loss_curve['train'], loss_curve['val'])),
                 win=plot_loss_win, opts=dict(title='Loss Curve', legend=['train', 'validation'])

                 )
        vis.line(X=np.column_stack((X, X)),
                 Y=np.column_stack((acc['train'], acc['val'])),
                 win=plot_acc, opts=dict(title='Accuracy', legend=['train', 'validation'])
                 )

        temp_time = time.time()
        time_remaining = (temp_time - since) / (epoch + 1) * (num_epochs - epoch - 1)
        print('\ttime remaining = {:.0f}m {:.0f}s'.format(time_remaining // 60, time_remaining % 60))

        print()
        if best_acc > 0.95:
            break
            #     plot_model(X,loss_curve, 'Loss', output_path)
            #     plot_model(X, acc, 'Accuracy', output_path)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if epoch < num_epochs - 1:
        print('Break at epoch{}'.format(epoch))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('At epoch: {}'.format(best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    stats = (loss_curve, acc, best_acc, best_epoch)

    return model, stats


def init_model(model):
    # model = models.alexnet(pretrained=True)
    # model.classifier[6] = nn.Linear(4096, 2)

    # # print(list(list(model.classifier.children())[1].parameters()))
    # mod = list(model.classifier.children())
    # mod.pop()
    # mod.append(torch.nn.Linear(4096, 2))
    # new_classifier = torch.nn.Sequential(*mod)
    # # print(list(list(new_classifier.children())[1].parameters()))
    # model.classifier = new_classifier
    # # m.classifier

    model = models.resnet18(progress=True, pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    #     torch.nn.init.xavier_uniform(model.fc.weight)

    count = 0
    for layer, child in model.named_children():
        count += 1
        #         print(layer)
        for name, param in child.named_parameters():
            if count < 5:
                param.requires_grad = False
            else:
                #                 param.requires_grad = True
                pass
                #                 print("\t",name)
    return model

def visualize_model(model, dataloader, device, output_path, num_images=8):
    was_training = model.training
    model.eval()
    images_so_far = 0
    #     fig = plt.figure()

    with torch.no_grad():

        for i, (inputs, labels, header) in enumerate(dataloader['val']):
            inputs = inputs.to(device, dtype=torch.float)
            labels = torch.from_numpy(np.asarray(labels, dtype='int64'))
            labels = labels.to(device)
            print('input.requires_grad =', inputs.requires_grad)
            # compute saliency map
            inputs.requires_grad = True
            torch.set_grad_enabled(True)
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            print(inputs.requires_grad)

            batch_size = inputs.size()[0]
            col = 2
            row = batch_size
            fig, ax = plt.subplots(row, col, figsize=(col * 2, row * 2))

            for j in range(inputs.size()[0]):
                images_so_far += 1

                inp = inputs.cpu().data[j]
                inp = inp.numpy().transpose((1, 2, 0))
                inp = rgb2gray(inp)
                ax[j, 0].imshow(inp, cmap=plt.cm.bone, vmin=0, vmax=1)
                ax[j, 0].axis('off')
                ax[j, 0].set_title('predicted: {}, label:{}'.format(preds[j], labels[j]))

                # ----
                outputs[j, preds[j]].backward(retain_graph=True)
                T = inputs.grad
                T = T.cpu().data[j]
                T = np.abs(T)
                T = T.numpy().transpose((1, 2, 0))
                T = rgb2gray(T)
                #                 T = T[T[0::][:,0:]>0.3]
                ma = T.max()
                mi = T.min()

                ax[j, 1].imshow(T, plt.get_cmap('hot'))
                ax[j, 1].axis('off')
                ax[j, 1].set_title('predicted: {}, label:{}'.format(preds[j], labels[j]))

                inputs.grad = None
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'saliency{}.png'.format(i)))
            if i == 2:
                return
        model.train(mode=was_training)




def parse_arguments():
    parser = argparse.ArgumentParser(description='Train model.')

    parser.add_argument('--lr', metavar='--learning_rate', type=float, default=1e-4,
                        help='leraning rate')
    parser.add_argument('--batch_size', metavar='--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--epoch', metavar='--epoch', type=int, default=100,
                        help='trainng epochs')
    parser.add_argument('--reg', metavar='--regularization', type=float, default=0,
                        help='L2 regularizaiton rate')
    parser.add_argument('--step_size', metavar='--step_size', type=int, default=10000,
                        help='learning rate decay step size')
    parser.add_argument('--gamma', metavar='--gamma', type=float, default=1,
                        help='learning rate decay rate')
    parser.add_argument('--dropout', metavar='--dropout', type=int, default=0,
                        help='dropout rate')
    parser.add_argument('--weight', metavar='--weight', type=float, default=1,
                        help='weight for loss')

    args = parser.parse_args()
    return args



