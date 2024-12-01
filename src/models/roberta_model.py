# pylint: disable=W0223
import pytorch_lightning as pl
import torch
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RoBERTa(pl.LightningModule):
    """
    Class for training RoBERTa via Pytorch Lightning.
    """

    def __init__(
        self,
        model_type: str=None,
        num_labels: int=None,
        tokenizer=None,
        steps_per_epoch: int=None,
        epochs: int=None,
        lr: float=3e-6,
        loss_fct_params: dict={},
        class_weights: List[int]=None,
        SUF_CHECK: bool=False,
        use_weighted_ce_loss: bool=True,
        label_smoothing: float=0.0,
    ):
        super().__init__()
        print(f"model_type: {model_type}")
        self.roberta = AutoModelForSequenceClassification.from_pretrained(
            model_type, num_labels=num_labels
        )
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_type, add_prefix_space=True
            )
        else:
            self.tokenizer = tokenizer

        self.model_type = model_type
        self.num_labels = num_labels

        self.class_weights = torch.FloatTensor(class_weights) if class_weights else torch.FloatTensor([1.0 for i in range(num_labels)])
        print(f"self.class_weights: {self.class_weights}")
        self.use_weighted_ce_loss = use_weighted_ce_loss
        print(f" self.use_weighted_ce_loss: {self.use_weighted_ce_loss}")


        self.roberta.config.hidden_dropout_prob = 0.1
        self.roberta.config.attention_probs_drop_prob = 0.1
        self.roberta.config.classifier_dropout = 0.1
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.label_smoothing = label_smoothing

        self.SUF_CHECK = SUF_CHECK
    
    def expand_embeddings(self):
        if (
            len(self.tokenizer)
            != self.roberta.roberta.embeddings.word_embeddings.weight.shape[0]
        ):
            print("Expanding embedding size")
            self.roberta.resize_token_embeddings(len(self.tokenizer))

    def forward(self, inputs, labels=None):  # pylint: disable=arguments-differ
        output = self.roberta(**inputs, labels=labels)
        return output

    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        if len(batch) == 3:
            x_batch, y_batch, suf_y_batch = batch

            if self.SUF_CHECK:
                y_label = suf_y_batch
            else:
                y_label = y_batch
        else:
             x_batch, y_label = batch

        output = self.forward(x_batch, y_label)
        y_hat = torch.argmax(output.logits, dim=1)

        if self.use_weighted_ce_loss:
            loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing, 
                                                 weight=self.class_weights.to(output.logits.device))
            loss = loss_fct(output.logits.view(-1, self.num_labels), y_label.view(-1))
        else:
            loss = output.loss

        accuracy = torch.sum(y_label == y_hat).item() / (len(y_label) * 1.0)
        print("train_loss", loss)
        print("train_accuracy", accuracy)

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        if len(batch) == 3:
            x_batch, y_batch, suf_y_batch = batch

            if self.SUF_CHECK:
                y_label = suf_y_batch
            else:
                y_label = y_batch
        else:
             x_batch, y_label = batch

        output = self.forward(x_batch, y_label)
        y_hat = torch.argmax(output.logits, dim=1)

        if self.use_weighted_ce_loss:
            loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing, 
                                                 weight=self.class_weights.to(output.logits.device))
            loss = loss_fct(output.logits.view(-1, self.num_labels), y_label.view(-1))
        else:
            loss = output.loss

        accuracy = torch.sum(y_label == y_hat).item() / (len(y_label) * 1.0)
        self.log("valid_loss", loss.detach())
        self.log("valid_accuracy", accuracy)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        print(f"self.epochs: {self.epochs}")
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.lr,
            pct_start=0.05,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            anneal_strategy="linear",
        )
        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]
