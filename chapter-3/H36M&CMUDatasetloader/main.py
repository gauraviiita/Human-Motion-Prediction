import torch
from torch.utils.data import DataLoader


from opt import Options
from h36motion3d import H36motion3D
import data_utils as data_utils


def main(opt):
    print('Hello Gaurav!')
    
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n
    sample_rate = opt.sample_rate

    # data loading
    print(">>> loading data")
    train_dataset = H36motion3D(path_to_data=opt.data_dir, 
                                actions=opt.actions, 
                                input_n=input_n, 
                                output_n=output_n,
                                split=0, 
                                dct_used=dct_n, 
                                sample_rate=sample_rate)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=opt.train_batch,
                              shuffle=True,
                              num_workers=opt.job,
                              pin_memory=True)

    val_dataset = H36motion3D(path_to_data=opt.data_dir, 
                              actions=opt.actions, 
                              input_n=input_n, 
                              output_n=output_n, 
                              split=2, 
                              dct_used=dct_n, 
                              sample_rate=sample_rate)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=opt.test_batch,
                            shuffle=False,
                            num_workers=opt.job,
                            pin_memory=True)

    acts = data_utils.define_actions('all')
    test_data = dict()
    for act in acts:
        test_dataset = H36motion3D(path_to_data=opt.data_dir, 
                                   actions=act, 
                                   input_n=input_n, 
                                   output_n=output_n, 
                                   split=1, 
                                   sample_rate=sample_rate, 
                                   dct_used=dct_n)
        test_data[act] = DataLoader(dataset=test_dataset, 
                                    batch_size=opt.test_batch, 
                                    shuffle=False, 
                                    num_workers=opt.job, 
                                    pin_memory=True)
    
    print(">>> data loaded !")
    print(">>> train data {}".format(train_dataset.__len__()))
    print(">>> test data {}".format(test_dataset.__len__()))
    print(">>> validation data {}".format(val_dataset.__len__()))

if __name__ == "__main__":
    option = Options().parse()
    main(option)
