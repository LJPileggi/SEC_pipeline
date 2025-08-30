import pickle
import torch

class CustomDataset(torch.utils.data.Dataset):
    """
    Container to handle the multiclass track dataset. Keeps track of track
    names for explainability purposes.
    
    args:
     - root: root directory containing the class directories;
     - device: torch device for tensor handling;
     - audio_paths: tensors of the audio tracks (the x-s of the dataset);
     - aufio_tracks: names of the tracks;
     - target: class labels of tracks;
     - classes: list of classes;
     - class_to_idx: indexing dictionary for classes.
    """
    def __init__(self, root, type_ds, device: str = 'cpu', map_class_together: dict = {}):
        """
        args:
         - root: root directory containing the class directories;
         - type_ds: the dataset split to load. Must lie among
           ['train', 'es', 'valid', 'test'];
         - device: torch device for tensor handling;
         - map_class_together: class mapper dictionary.
        """
        super().__init__()
        self.root = root
        self.device = device
        self.targets = []
        self.audio_paths = []
        self.audio_tracks = []
        classes = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
        map_class = {}
        for k, v in map_class_together.items():
            for c in v:
                map_class[c] = k
        for i, class_ in enumerate(classes):
            class_dir = os.path.join(root, class_, type_ds)
            for fp in tqdm(os.listdir(class_dir), desc=f'{i+1}/{len(classes)} class {class_}'):
                fullpath_audio = os.path.join(class_dir, fp)
                try:
                    self.audio_tracks.append(fullpath_audio)
                    self.audio_paths.append(torch.load(fullpath_audio))
                    self.targets.append(map_class[class_] if class_ in map_class else class_)
                except:
                    pass
        self.classes = sorted(list(set(self.targets)))
        self.class_to_idx = {v: k for k, v in enumerate(self.classes)}


    def __getitem__(self, index):
        """
        Yields tracks and their labels as (x, y) couples.
        """
        x, target = self.audio_paths[index], self.targets[index]
        y = torch.zeros(len(self.classes))
        y[self.class_to_idx[target]] = 1
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    def __len__(self):
        return len(self.audio_paths)

    def class_percentage_balancement(self):
        return {i: self.targets.count(i) for i in self.classes}


def create_dataset(root, type_ds, *args, **kwargs):
    """
    Generates the given dataset split from given directory as
    a CustomDataset object. If already created, it loads it from
    a binary file, otherwise it creates it from scratch from the
    track embeddings.
    
    args:
     - root: root directory containing the class directories for embeddings;
     - type_ds: the dataset split to load. Must lie among
       ['train', 'es', 'valid', 'test'];
     - *args: further CustomDataset args;
     - **kwargs: CustomDataset kwargs
    
    returns:
     - ds: CustomDataset containing the desired embedding dataset split.
    """
    main_ds_pickle_path = os.path.join(root, f'dataset.pickle')
    ds_pickle_path = os.path.join(root, f'dataset_{type_ds}.pickle')
    if os.path.exists(main_ds_pickle_path):
        os.rename(main_ds_pickle_path, ds_pickle_path)
    print(f'CREATE DATASET => {root.split("/")[-1]}_{type_ds}')
    if os.path.exists(ds_pickle_path):
        ds = pkl.load(open(ds_pickle_path, 'rb'))
    else:
        ds = CustomDataset(
                root=root,
                type_ds=type_ds,
                *args,
                **kwargs
        )
        pkl.dump(ds, open(ds_pickle_path, 'wb'))
    return ds

def data_stats(dataset):
    """
    Yields information about the balancing of the given dataset.
    
    args:
     - dataset: CustomDataset for embeddings.
    """
    blc = dataset.class_percentage_balancement()
    print(f'class balancement: {blc}')
    list_blc = np.array(list(blc.values()))
    print(f'stats: min={np.min(list_blc)}, max={np.max(list_blc)}, ' \
          f'mean={np.mean(list_blc)}, std={np.std(list_blc)}')

def load_octaveband_embeddings(octaveband_dir, batch_size):
    """
    Generates lists of DataLoaders and CustomDatasets for embeddings
    of a given octaveband run.
    
    args:
     - octaveband_dir: directory for a given octaveband run;
     - batch_size: batch size for DataLoaders.

    returns:
     - dataloaders: list of DataLoader objects containing the datasets;
     - custom_data: list of CustomDataset objects containing the datasets;
    """
    dataloaders = {fp: [torch.utils.data.DataLoader(create_dataset(
        root=os.path.join(octaveband_dir, fp),
        type_ds=type_ds,
        device = device,
        map_class_together = {}
    ), batch_size=batch_size, shuffle=True) for type_ds in ['train', 'es', 'valid', 'test']] for fp in os.listdir(octaveband_dir)}

    custom_data = {fp: [create_dataset(
        root=os.path.join(octaveband_dir, fp),
        type_ds=type_ds,
        device = device,
        map_class_together = {}
    ) for type_ds in ['train', 'es', 'valid', 'test']] for fp in os.listdir(octaveband_dir)}
    return dataloaders, custom_data

