import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

def conv_len(lin, pad, dil, ker, stride):
    return int((lin + 2 * pad - dil * (ker - 1) - 1) / stride + 1)

def pool_len(lin, pad, ker, stride):
    return int((lin + 2 * pad - ker) / stride + 1)

class AbnormalBreathDetectorCNN(pl.LightningModule):
    def __init__(self, learning_rate, k_sizes, dilatations, cout, fixed_len, dropout, pool_kern=None):
        super(AbnormalBreathDetectorCNN, self).__init__()
        self.save_hyperparameters()
        self.k_sizes = k_sizes
        self.dilatations = dilatations
        self.c_out_n = cout
        self.fixed_len = fixed_len
        self.dropout = torch.nn.Dropout(dropout)
        self.pool_kern = pool_kern
        self.conv_modules = nn.ModuleList([nn.Conv1d(2, self.c_out_n, k, dilation=d) for d in self.dilatations for k in self.k_sizes])
        if self.pool_kern is not None:
            self.pooling = nn.AvgPool1d(self.pool_kern)
            self.out_sizes = [pool_len(conv_len(self.fixed_len, 0, d, k, 1), 0, self.pool_kern, self.pool_kern) for d in self.dilatations for k in self.k_sizes]
        else:
            self.out_sizes = [conv_len(self.fixed_len, 0, d, k, 1) for d in self.dilatations for k in self.k_sizes]
        self.fc = torch.nn.Linear(sum(self.out_sizes) * self.c_out_n, 1)
        self.relu = torch.nn.ReLU()
        self.lr = learning_rate
        self.epoch = 0
        
    def forward(self, xs):
        if self.pool_kern is not None:
            convs = [self.dropout(self.pooling(self.relu(m(xs)))) for m in self.conv_modules]
        else:
            convs = [self.dropout(self.relu(m(xs))) for m in self.conv_modules]
        convs = torch.cat([convs[i].view(-1, self.out_sizes[i] * self.c_out_n) for i in range(len(self.out_sizes))], dim=1)
        out = self.fc(convs)
        pred = torch.sigmoid(out)
        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
        return {"optimizer" : optimizer, "lr_scheduler" : lr_scheduler, "monitor" : "Validation_F1_log"}
    
    def training_step(self, batch, batch_idx):
        _, x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat.flatten(), y)
        return {"loss" : loss, "preds" : y_hat.flatten(), "labels" : y}
    
    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        preds = torch.cat([x["preds"] for x in training_step_outputs])
        targets = torch.cat([x["labels"] for x in training_step_outputs])
        print("Epoch {} train".format(self.epoch))
        confmat = torchmetrics.ConfusionMatrix(num_classes=2)
        print(confmat(preds.cpu(), targets.int().cpu()))
        f1score = torchmetrics.F1Score(num_classes=1)
        f1_epoch = f1score(preds.cpu(), targets.int().cpu())
        recall = torchmetrics.Recall(num_classes=1)
        recall_epoch = recall(preds.cpu(), targets.int().cpu())
        specificity = torchmetrics.Specificity(num_classes=1)
        specificity_epoch = specificity(preds.cpu(), targets.int().cpu())
        auroc = torchmetrics.AUROC(pos_label=1)
        auroc_epoch = auroc(preds.cpu(), targets.int().cpu())
        self.logger.experiment.add_scalar("Train_Loss", avg_loss.item(), self.epoch)
        self.logger.experiment.add_scalar("Train_F1", f1_epoch, self.epoch)
        self.logger.experiment.add_scalar("Train_recall", recall_epoch, self.epoch)
        self.logger.experiment.add_scalar("Train_specificity", specificity_epoch, self.epoch)
        self.logger.experiment.add_scalar("Train_AUROC", auroc_epoch, self.epoch)
    
    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat.flatten(), y)
        return {"loss" : loss, "preds" : y_hat.flatten(), "labels" : y}
    
    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        preds = torch.cat([x["preds"] for x in validation_step_outputs])
        targets = torch.cat([x["labels"] for x in validation_step_outputs])
        print("Epoch {} validation".format(self.epoch))
        confmat = torchmetrics.ConfusionMatrix(num_classes=2)
        print(confmat(preds.cpu(), targets.int().cpu()))
        f1score = torchmetrics.F1Score(num_classes=1)
        f1_epoch = f1score(preds.cpu(), targets.int().cpu())
        recall = torchmetrics.Recall(num_classes=1)
        recall_epoch = recall(preds.cpu(), targets.int().cpu())
        specificity = torchmetrics.Specificity(num_classes=1)
        specificity_epoch = specificity(preds.cpu(), targets.int().cpu())
        auroc = torchmetrics.AUROC(pos_label=1)
        auroc_epoch = auroc(preds.cpu(), targets.int().cpu())
        self.logger.experiment.add_scalar("Validation_Loss", avg_loss.item(), self.epoch)
        self.logger.experiment.add_scalar("Validation_F1", f1_epoch, self.epoch)
        self.logger.experiment.add_scalar("Validation_recall", recall_epoch, self.epoch)
        self.logger.experiment.add_scalar("Validation_specificity", specificity_epoch, self.epoch)
        self.logger.experiment.add_scalar("Validation_AUROC", auroc_epoch, self.epoch)
        self.log("Validation_F1_log", f1_epoch)
        self.epoch += 1
    
    def test_step(self, batch, batch_idx):
        _, x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat.flatten(), y)
        return {"loss" : loss, "preds" : y_hat.flatten(), "labels" : y}
    
    def test_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x["loss"] for x in validation_step_outputs]).mean()
        preds = torch.cat([x["preds"] for x in validation_step_outputs])
        targets = torch.cat([x["labels"] for x in validation_step_outputs])
        print("Test Results")
        confmat = torchmetrics.ConfusionMatrix(num_classes=2)
        print(confmat(preds.cpu(), targets.int().cpu()))
        f1score = torchmetrics.F1Score(num_classes=1)
        f1_epoch = f1score(preds.cpu(), targets.int().cpu())
        recall = torchmetrics.Recall(num_classes=1)
        recall_epoch = recall(preds.cpu(), targets.int().cpu())
        specificity = torchmetrics.Specificity(num_classes=1)
        specificity_epoch = specificity(preds.cpu(), targets.int().cpu())
        auroc = torchmetrics.AUROC(pos_label=1)
        auroc_epoch = auroc(preds.cpu(), targets.int().cpu())
        self.logger.experiment.add_scalar("Test_Loss", avg_loss.item(), self.epoch)
        self.logger.experiment.add_scalar("Test_F1", f1_epoch, self.epoch)
        self.logger.experiment.add_scalar("Test_Recall", recall_epoch, self.epoch)
        self.logger.experiment.add_scalar("Test_Specificity", specificity_epoch, self.epoch)
        self.logger.experiment.add_scalar("Test_AUROC", auroc_epoch, self.epoch)