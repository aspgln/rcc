from __future__ import print_function, division


import matplotlib.pyplot as plt
import numpy as np

def show_batch(sample_batched):
    """show image with label for a batch of samples"""
    image_batch, lable_batch = sample_batched[0], sample_batched[1]

    batch_size = len(image_batch)
    im_size = image_batch.size(2)

    col = 4
    row = int((batch_size) / col)

    fig, ax = plt.subplots(nrows=row, ncols=col)
    for i in range(row):
        for j in range(col):
            print('batch no ', i*4+j)
            #             im = ax[i,j].imshow(image_batch[i*4+j, 1, :, :], cmap=plt.cm.bone, vmin = -360, vmax = 440)

            if row > 1:
                # show RGB
                im = ax[i, j].imshow((np.transpose(image_batch[i * 4 + j, :, :, :])+2)/4)


                ax[i, j].axis('off')
                ax[i, j].set_title(lable_batch[i * 4 + j].numpy())
            if row == 1:
                im = ax[j].imshow((np.transpose(image_batch[i * 4 + j, :, :, :])+2)/4)
                ax[j].axis('off')
                ax[j].set_title(lable_batch[i * 4 + j].numpy())
    cb_ax = fig.add_axes([0.96, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    # cbar = fig.colorbar(im)

    plt.show()
    return


def show_batch2(sample_batched, vmin=-360, vmax=440):
    """show image with label for a batch of samples"""
    image_batch, lable_batch = sample_batched[0], sample_batched[1]
    batch_size = len(image_batch)
    im_size = image_batch.size(2)

    col = 4
    row = int((batch_size) / col)
    # fig,ax = plt.subplots(nrows=row)
    for i in range (row):
        fig, ax = plt.subplots()

        im1 = image_batch[i*4+0, 0, :, :]
        im2 = image_batch[i*4+1, 0, :, :]
        im3 = image_batch[i*4+2, 0, :, :]
        im4 = image_batch[i*4+3, 0, :, :]

        im_batch = np.hstack((im1,im2,im3,im4))
        im = ax.imshow(im_batch, cmap=plt.cm.bone, vmin=vmin, vmax=vmax)

        cb_ax = fig.add_axes([0.96, 0.1, 0.02, 0.5])
        cbar = fig.colorbar(im, cax=cb_ax)
        # cbar = fig.colorbar(im)
        plt.show()
    # plt.show()
    return

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    (tn, fp, fn, tp) = cm.ravel()
    print('p/t     1     0')
    print('  1    {}    {}'.format(tp, fp))
    print('  0     {}    {}'.format(fn, tn))


def evaluate(pred_list, label_list):
    tp_idx = []
    fp_idx = []
    fn_idx = []
    tn_idx = []
    for i in range(len(label_list)):
    #     print(la[i])
        if label_list[i] == 1 and pred_list[i] == 1:
            tp_idx.append(i)
        elif label_list[i] == 0 and pred_list[i] == 0:
            tn_idx.append(i)
        elif label_list[i] == 1 and pred_list[i] == 0:
            fn_idx.append(i)
        elif label_list[i] == 0 and pred_list[i] == 1:
            fp_idx.append(i)

    sens = len(tp_idx)/(len(tp_idx)+len(fn_idx))
    spec = len(tn_idx)/(len(tn_idx)+len(fp_idx))
    acc = (len(tn_idx) + len(tp_idx) )/(len(tn_idx) + len(tp_idx)+len(fn_idx) + len(fp_idx))
    return sens,spec,acc