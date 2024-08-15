import os
import pathlib
import sklearn.model_selection
import sys
import torch

here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..' / '..'))

import controldiffeq


def dataloader(dataset, **kwargs):
    if 'shuffle' not in kwargs:
        kwargs['shuffle'] = True
    if 'drop_last' not in kwargs:
        kwargs['drop_last'] = True
    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 32
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = 8
    kwargs['batch_size'] = min(kwargs['batch_size'], len(dataset))
    return torch.utils.data.DataLoader(dataset, **kwargs)


def split_data(tensor, stratify):
    # 0.7/0.15/0.15 train/val/test split
    (train_tensor, testval_tensor,
     train_stratify, testval_stratify) = sklearn.model_selection.train_test_split(tensor, stratify,
                                                                                  train_size=0.7,
                                                                                  random_state=0,
                                                                                  shuffle=True,
                                                                                  stratify=stratify)

    val_tensor, test_tensor = sklearn.model_selection.train_test_split(testval_tensor,
                                                                       train_size=0.5,
                                                                       random_state=1,
                                                                       shuffle=True,
                                                                       stratify=testval_stratify)
    return train_tensor, val_tensor, test_tensor


def normalise_data(X, y):
    train_X, _, _ = split_data(X, y)
    out = []
    # import pdb;pdb.set_trace()
    for Xi, train_Xi in zip(X.unbind(dim=-1), train_X.unbind(dim=-1)):
        train_Xi_nonan = train_Xi.masked_select(~torch.isnan(train_Xi))
        m = train_Xi_nonan.min()
        M = train_Xi_nonan.max()
        out.append((Xi-m)/(M-m))
        # mean = train_Xi_nonan.mean()  # compute statistics using only training data.
        # std = train_Xi_nonan.std()
        # out.append((Xi - mean) / (std + 1e-5))
    out = torch.stack(out, dim=-1)
    return out


def preprocess_data(times, X, y, final_index, append_times, append_intensity):
    X = normalise_data(X, y)

    # Append extra channels together. Note that the order here: time, intensity, original, is important, and some models
    # depend on that order.

    # augmented_X = []
    # if append_times:
    #     augmented_X.append(times.unsqueeze(0).repeat(X.size(0), 1).unsqueeze(-1))
    # if append_intensity:
    #     intensity = ~torch.isnan(X)  # of size (batch, stream, channels)
    #     intensity = intensity.to(X.dtype).cumsum(dim=1)
    #     augmented_X.append(intensity)
    # augmented_X.append(X)
    # if len(augmented_X) == 1:
    #     X = augmented_X[0]
    # else:
    #     X = torch.cat(augmented_X, dim=2)

    train_X, val_X, test_X = split_data(X, y)
    train_y, val_y, test_y = split_data(y, y)
    train_final_index, val_final_index, test_final_index = split_data(final_index, y)
    print('train')
    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, train_X)
    print('val')
    val_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, val_X)
    print('test')
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, test_X)

    in_channels = X.size(-1)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, in_channels, train_X, val_X, test_X)


def wrap_data(times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
              test_final_index, device, batch_size, train_X,val_X,test_X, num_workers=4):
    # import pdb;pdb.set_trace()
    times = times.to(device)
    train_coeffs = tuple(coeff.to(device) for coeff in train_coeffs)
    val_coeffs = tuple(coeff.to(device) for coeff in val_coeffs)
    test_coeffs = tuple(coeff.to(device) for coeff in test_coeffs)
    train_y = train_y.to(device)
    val_y = val_y.to(device)
    test_y = test_y.to(device)
    train_final_index = train_final_index.to(device)
    val_final_index = val_final_index.to(device)
    test_final_index = test_final_index.to(device)
    train_X=train_X.to(device)
    val_X=val_X.to(device)
    test_X=test_X.to(device)
    train_idx = torch.tensor(list(range(len(train_y)))).to(device)
    class_num = train_y.max()+1
    data_size = train_X.shape[0]
    seq_len = train_X.shape[1]

    rest_X = torch.cat((test_X,val_X))
    rest_final_index = torch.cat((test_final_index,val_final_index))
    rest_y = torch.cat((test_y,val_y))

    # import pdb;pdb.set_trace()
    class_train_y = torch.eye(class_num)[train_y].view(data_size,1,class_num).repeat(1,seq_len,1).to(device)
    aug_train_X = torch.cat((train_X,class_train_y),dim=2)
    data_size = rest_X.shape[0]
    # import pdb;pdb.set_trace()
    class_rest_y = torch.eye(class_num)[rest_y].view(data_size,1,class_num).repeat(1,seq_len,1).to(device)
    aug_rest_X = torch.cat((rest_X,class_rest_y),dim=2)
    # class_train_y = torch.eye(class_num)[train_y].view(data_size,1,class_num).repeat(1,seq_len,1).to(device)
    # aug_train_X = torch.cat((train_X,class_train_y),dim=2)
    # tmp = 244
    # import pdb;pdb.set_trace()
    # a,b,c,d = train_coeffs
    # a = torch.cat([a[0:tmp],a[tmp+1:]])
    # b = torch.cat([b[0:tmp],b[tmp+1:]])
    # c = torch.cat([c[0:tmp],c[tmp+1:]])
    # d = torch.cat([d[0:tmp],d[tmp+1:]])
    # train_coeffs = tuple([a,b,c,d])
    # train_y = torch.cat([train_y[0:tmp],train_y[tmp+1:]])
    # train_final_index = torch.cat([train_final_index[0:tmp],train_final_index[tmp+1:]])
    # train_X = torch.cat([train_X[0:tmp],train_X[tmp+1:]])
    # train_idx = torch.cat([train_idx[0:tmp],train_idx[tmp+1:]])
    
    train_dataset = torch.utils.data.TensorDataset(*train_coeffs, train_y, train_final_index,aug_train_X,train_idx)
    val_dataset = torch.utils.data.TensorDataset(*val_coeffs, val_y, val_final_index,val_X)
    test_dataset = torch.utils.data.TensorDataset(rest_y, rest_final_index,aug_rest_X)

    train_dataloader = dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return times, train_dataloader, val_dataloader, test_dataloader, train_dataset, val_dataset, test_dataset


def save_data(dir, **tensors):
    for tensor_name, tensor_value in tensors.items():
        torch.save(tensor_value, str(dir / tensor_name) + '.pt')


def load_data(dir):
    tensors = {}
    for filename in os.listdir(dir):
        if filename.endswith('.pt'):
            tensor_name = filename.split('.')[0]
            tensor_value = torch.load(str(dir / filename))
            tensors[tensor_name] = tensor_value
    return tensors
