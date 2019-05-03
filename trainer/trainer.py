import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, data in enumerate(self.data_loader):
            data = data.to(self.device).type(torch.float)

            self.optimizer.zero_grad()
            output = self.model(data)
            kl, nll = self.loss(output, data)
            loss = kl + nll
            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, data)

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    loss.item()))
                # set step
                self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
                # add losses
                self.writer.add_scalar('loss', loss.item())
                self.writer.add_scalar('kl', kl.item())
                self.writer.add_scalar('nll', nll.item())
                # add gating params
                # tag = "gates/{0}/{1}"
                # for i, p in enumerate(self.model.gating_params):
                #     for j, v in enumerate(p.data):
                #         self.writer.add_scalar(tag.format(i, i+j+1), v.item())

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # decay temperature by 0.99 every epoch
        self.model.tau *= 0.99
        print("tau decayed to %f" % self.model.tau)

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                data = data.to(self.device)

                output = self.model(data)
                kl, nll = self.loss(output, data)
                loss = kl + nll

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                self.writer.add_scalar('kl', kl.item())
                self.writer.add_scalar('nll', nll.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, data)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
