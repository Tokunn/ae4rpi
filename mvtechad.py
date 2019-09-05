from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm


IMGSIZE = 128

mvtechad_path = os.path.expanduser('~/group/msuzuki/MVTechAD')
capsule_path = os.path.join(mvtechad_path, 'capsule')
train_path = os.path.join(capsule_path, 'train')
test_path = os.path.join(capsule_path, 'test')
ground_truth_path = os.path.join(capsule_path, 'ground_truth')

npy_path = './npy'
os.makedirs(npy_path, exist_ok=True)

def get_imglist(path):
    # Get directory list
    path = os.path.join(path, '*')
    dirlist = sorted(glob.glob(path))
    dirlist = [p for p in dirlist if os.path.isdir(p)]
    # Get image list
    imglist = []
    for d in dirlist:
        dirpath = os.path.join(d, '*.png')
        imgs = sorted(glob.glob(dirpath))
        imglist += imgs
    print("{} images".format(len(imglist)))
    return imglist

def load_imgs(path, imgsize, ground=None):
    imgnamelist = get_imglist(path)
    goodimgnamelist = [c for c in imgnamelist if 'good' in c] 
    badimgnamelist = [c for c in imgnamelist if not 'good' in c]

    goodimglist = []
    for imgname in tqdm(goodimgnamelist):
        img = Image.open(imgname)
        resized_img = img.resize((imgsize, imgsize))
        goodimglist.append(np.array(resized_img))

    badimglist = []
    for imgname in tqdm(badimgnamelist):
        img = Image.open(imgname)
        resized_img = img.resize((imgsize, imgsize))
        badimglist.append(np.array(resized_img))

    y = np.asarray([])
    if ground is not None:
        truthimglist = []
        truthimgnamelist = get_imglist(ground)
        for imgname in tqdm(truthimgnamelist):
            img = Image.open(imgname)
            resized_img = img.resize((imgsize, imgsize))
            truthimglist.append(np.array(resized_img))
        truthimglist = np.array(truthimglist)
        zeros = np.zeros((len(goodimglist), truthimglist.shape[1], truthimglist.shape[2]))
        y = np.concatenate((truthimglist, zeros))

    x = np.asarray(badimglist+goodimglist)
    return x, y


def load_data():
    print("Loading test data...") 
    try:
        x_test = np.load(os.path.join(npy_path, 'x_test.npy'))
        y_test = np.load(os.path.join(npy_path, 'y_test.npy'))
    except FileNotFoundError:
        x_test, y_test = load_imgs(test_path, IMGSIZE, ground=ground_truth_path)
        np.save(os.path.join(npy_path, 'x_test.npy'), x_test)
        np.save(os.path.join(npy_path, 'y_test.npy'), y_test)

    print("Loading train data...")
    try:
        x_train = np.load(os.path.join(npy_path, 'x_train.npy'))
        y_train = x_train
    except FileNotFoundError:
        x_train, _ = load_imgs(train_path, IMGSIZE)
        y_train = x_train
        np.save(os.path.join(npy_path, 'x_train.npy'), x_train)

    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
