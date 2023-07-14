class Net_Large(pl.LightningModule):
    def __init__(self):
        super(Net_Large, self).__init__()
        
        self.conv1 = GATConv(1, 32, heads=4)
        # self.lin1 = torch.nn.Linear(1, 4 * 32)
        self.conv2 = GATConv(4 * 32, 32, heads=4)
        # self.lin2 = torch.nn.Linear(4 * 32, 4 * 32)
        self.conv3 = GATConv(4 * 32, 32, heads=8)
        # self.lin3 = torch.nn.Linear(4 * 32, 8 * 32)
        self.conv4 = GATConv(8 * 32, len(class_weights), heads=6,concat=False)
        # self.lin4 = torch.nn.Linear(8 * 32, len(class_weights))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        
        
#         x = F.elu(self.conv1(x, edge_index, edge_weight) + self.lin1(x))
#         x = F.elu(self.conv2(x, edge_index, edge_weight) + self.lin2(x))
#         x = F.elu(self.conv3(x, edge_index, edge_weight) + self.lin3(x))
#         x = self.conv4(x, edge_index, edge_weight) + self.lin4(x)
        
#         return x

        x = F.elu(self.conv1(x, edge_index, edge_weight)) #+ self.lin1(x)
        x = F.elu(self.conv2(x, edge_index, edge_weight))
        x = F.elu(self.conv3(x, edge_index, edge_weight))
        x = self.conv4(x, edge_index, edge_weight)
        return x

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        return optimizer
    
    def training_step(self, data, batch_idx): 
        out = model(data)
        loss_function = CrossEntropyLoss(weight=class_weights).to(device)
        
        train_loss = loss_function(out[data.train_mask], data.y[data.train_mask].squeeze().to(torch.int64))
        
        correct=out[data.train_mask].argmax(dim=1).eq(data.y[data.train_mask]).sum().item()
        logs={"train_loss": train_loss}
        total=len(data.y[data.train_mask])
        batch_dictionary={"loss": train_loss, "log": logs, "correct": correct, "total": total}
        return train_loss
    
#     def validation_step(self, data, batch_idx):

#         out = model(data)
#         loss_function = CrossEntropyLoss(weight=class_weights).to(device) #weight=class_weight
#         val_loss = loss_function(out[data.val_mask], data.y[data.val_mask].squeeze().to(torch.int64))

#         ys, preds = [], []
#         val_label = data.y[data.val_mask].cpu()
#         ys.append(data.y[data.val_mask])
#         preds.append((out[data.val_mask].argmax(-1)).float().cpu())     
#         y, pred = torch.cat(ys, dim=0), torch.cat(preds, dim=0)
#         pred = pred.reshape(-1,1)
#         accuracy = (pred == val_label).sum() / pred.shape[0]

#         self.log("val_loss", val_loss)
#         self.log("val_acc", accuracy)

    def test_step(self, data, batch_idx):
        # this is the test loop
        out = model(data)
        loss_function = CrossEntropyLoss(weight=class_weights).to(device) #weight=class_weight
        test_loss = loss_function(out[data.test_mask], data.y[data.test_mask].squeeze().to(torch.int64))
        
        ys, preds = [], []
        test_label = data.y[data.test_mask].cpu()
        ys.append(data.y[data.test_mask])
        preds.append((out[data.test_mask].argmax(-1)).float().cpu())

        y, pred = torch.cat(ys, dim=0), torch.cat(preds, dim=0)
        pred = pred.reshape(-1,1)
        accuracy = (pred == test_label).sum() / pred.shape[0]
        
        self.log("test_acc", accuracy)
        return pred, y.squeeze()
        
    def test_epoch_end(self, outputs):
        #this function gives us in the outputs all acumulated pred and test_labels we returned in test_step
        #we transform the pred and test_label into a shape that the classification report can read
        global true_array, pred_array
        true_array=[outputs[i][1].cpu().numpy() for i in range(len(outputs))]
        pred_array = [outputs[i][0].cpu().numpy() for i in range(len(outputs))]
        pred_array = np.concatenate(pred_array, axis=0 )
        true_array = np.concatenate(true_array, axis=0 )
        print(confusion_matrix(true_array, pred_array))
        print(classification_report(true_array, pred_array))