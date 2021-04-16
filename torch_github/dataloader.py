from torch.utils.data import DataLoader

class JavaCodeDataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(JavaCodeDataloader, self).__init__(*args, **kwargs)