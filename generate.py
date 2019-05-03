import os
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
import torch.distributions as D
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def main(config, resume):
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = get_instance(module_arch, 'arch', config)

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    unit_normal = D.MultivariateNormal(torch.zeros(5), torch.eye(5))
    generation_batch_size = 100
    
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(data_loader)):
#             data = data.to(device)
#             output = model(data)
#             output = torch.sigmoid(output['mu'])
#             output = output.reshape(-1,28,28).detach().cpu().numpy()
#             fig = plt.figure()
#             for i in range(20):
#                 img = output[i]
# #         img = (img > 0.5*np.max(img)+0.5*np.min(img)).astype(float) * 255
#                 plt.imshow(img, cmap = 'Greys', interpolation = 'nearest')
#                 plt.savefig('image{}.png'.format(i))
#             break
    with torch.no_grad():
        z = unit_normal.sample_n(generation_batch_size).to(device) #Batch Size X 5
        output = model.decoder(z)
        output = torch.sigmoid(output)
        output = output.reshape(generation_batch_size,28,28).detach().cpu().numpy()
        fig = plt.figure()
        for i in range(20):
            img = output[i]
#         img = (img > 0.5*np.max(img)+0.5*np.min(img)).astype(float) * 255
            plt.imshow(img, cmap = 'Greys', interpolation = 'nearest')
            plt.savefig('image{}.png'.format(i))
        np.savetxt('images.txt',output[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)

