from tqdm import tqdm
import torch

_criterion = torch.nn.CrossEntropyLoss()

def accuracy(y, p):
    return ((y.argmax(axis=-1) - p.argmax(axis=-1)) == 0).to(torch.float).mean().item()


def get_scores(model, test_dataloader, train_dataloader=None):
    """
    Yields loss, accuracy and confusion matrix on test set,
    and training set if the training set dataloader is given.
    
    args:
     - model: torch model of which to get the scores;
     - test_dataloader: dataloader for the test set;
     - train_dataloader (optional): dataloader for the training set;
     
    returns:
     - tr_loss (only if train_dataloader != None): loss on training set;
     - ts_loss: loss on test set;
     - tr_accuracy (only if train_dataloader != None): accuracy on training set;
     - ts_accuracy: accuracy on test set;
     - cm: confusion matrix on test set.
    """
    tr_loss, ts_loss = 0, 0
    tr_accuracy, ts_accuracy= 0, 0
    try:
        model.eval()
    except:
        pass
    p_list, y_list = [], []
    if train_dataloader is not None:
        for x, y in tqdm(train_dataloader, desc=f'loss on train'):
            p = model(x)
            tr_loss += _criterion(p, y).item()
            tr_accuracy += accuracy(y, p)
    for x, y in tqdm(test_dataloader, desc=f'loss on test'):
        p = model(x)
        p_list += p.argmax(axis=-1).tolist()
        y_list += y.argmax(axis=-1).tolist()
        ts_loss += _criterion(p, y).item()
        ts_accuracy += accuracy(y, p)
    if train_dataloader is not None:
        tr_loss = tr_loss / len(train_dataloader)
        tr_accuracy = tr_accuracy / len(train_dataloader)
    ts_loss = ts_loss / len(test_dataloader)
    ts_accuracy = ts_accuracy / len(test_dataloader)
    print(f'TEST => accuracy: {ts_accuracy} - loss: {ts_loss}')
    classes = test_dataloader.dataset.classes
    y_list = [classes[v] for v in  y_list]
    p_list = [classes[v] for v in  p_list]
    cm = confusion_matrix(y_list, p_list, labels=classes)
    if train_dataloader is None:
        return ts_loss, ts_accuracy, cm
    print(f'TRAIN => accuracy: {tr_accuracy} - loss: {tr_loss}')
    return tr_loss, ts_loss, tr_accuracy, ts_accuracy, cm
