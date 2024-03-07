import torch
import logging, coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install('DEBUG', logger=logger)

class ModelController:
    def __init__(self, model, loss_function, optimizer, model_train_epoch_count):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_train_epoch_count = model_train_epoch_count

    def train_model(self, data_loader):
        num_batches = len(data_loader)
        total_loss = 0
        self.model.train()

        for X, y in data_loader:
            output = self.model(X)
            loss = self.loss_function(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    
    def test_model(self, data_loader):
        num_batches = len(data_loader)
        total_loss = 0

        self.model.eval()
        with torch.no_grad():
            for X, y in data_loader:
                output = self.model(X)
                total_loss += self.loss_function(output, y).item()

        avg_loss = total_loss / num_batches
        return avg_loss

    
    def train_test_model(self, train_loader, test_loader, model_filename):
        logger.debug(f'[modelcontroller:train_test_model] train_loader : {train_loader}')
        # clear_output(wait=True)
        train_loss_values = []
        test_loss_values = []

        for ix_epoch in range(self.model_train_epoch_count):
            if ix_epoch % 10 == 0:
                logger.debug(f'[modelcontroller:test_train_model] start processing epoch {ix_epoch}')
            train_loss = self.train_model(train_loader)
            test_loss = self.test_model(test_loader)
            train_loss_values.append(train_loss)
            test_loss_values.append(test_loss)
            if ix_epoch %10 == 0:
                logger.debug(f'[modelcontroller:test_train_model] train loss : {train_loss}')
                logger.debug(f"[modelcontroller:test_train_model] test loss  : {test_loss}")
                logger.debug('---------------------')

        torch.save(self.model.state_dict(), model_filename)
        return train_loss_values, test_loss_values

    def predict(self, data_loader):
        output = torch.tensor([])
        self.model.eval()
        with torch.no_grad():
            for X, _ in data_loader:
                y_star = self.model(X)
                output = torch.cat((output, y_star), 0)

        return output.numpy()

