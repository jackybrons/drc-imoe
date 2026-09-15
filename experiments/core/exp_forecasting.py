from experiments.core.exp_basic import Exp_Basic
from datasets.loader import *
import torch
import torch.nn as nn
from torch import optim
import os
import json
import time
import warnings
import numpy as np
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.artifact_paths import portable_artifact_path
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.colors import LinearSegmentedColormap
import time
def cv_squared(x):
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.tensor([0], device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean() ** 2 + eps)


def _global_metrics(prediction, true_values):
    prediction = np.asarray(prediction)
    true_values = np.asarray(true_values)
    difference = prediction - true_values
    rmse = np.sqrt(np.mean(difference ** 2))
    mae = np.mean(np.abs(difference))
    denominator = np.where(
        np.abs(true_values) < 1e-8,
        np.nan,
        np.abs(true_values),
    )
    mape = np.nanmean(np.abs(difference / denominator)) * 100.0
    centered = true_values - np.mean(true_values)
    r2 = 1.0 - np.sum(difference ** 2) / np.sum(centered ** 2)
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape_percent': float(mape),
        'r2': float(r2),
    }

class Exp_Long_Term_Forecast1(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast1, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        return model

    def _get_data(self):
        if self.args.dataset == 'UL-NCA':
            train_loader, X_test_tensor, y_test_tensor, scaler = NCA_trainloader(self.args)
        elif self.args.dataset == 'UL-NCM':
            train_loader, X_test_tensor, y_test_tensor, scaler = NCM_trainloader(self.args)
        elif self.args.dataset == 'UL-NCMNCA':
            train_loader, X_test_tensor, y_test_tensor, scaler = NCMNCA_trainloader(self.args)
        elif self.args.dataset == 'TPSL':
            train_loader, X_test_tensor, y_test_tensor, scaler = TPSL_trainloader(self.args)
        elif self.args.dataset == 'LSD':
            train_loader, X_test_tensor, y_test_tensor, scaler = LSD_trainloader(self.args)
        return train_loader, X_test_tensor, y_test_tensor,scaler

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        legacy_prefixes = {
            'capacity_fcs.': 'capacity_fc.',
            'relaxation_fc.': 'features_fc.',
        }
        remapped_state_dict = {}
        for key, value in state_dict.items():
            mapped_key = key
            for legacy_prefix, current_prefix in legacy_prefixes.items():
                if key.startswith(legacy_prefix):
                    mapped_key = current_prefix + key[len(legacy_prefix):]
                    break
            if mapped_key in remapped_state_dict:
                raise ValueError(f'Duplicate checkpoint key after remapping: {mapped_key}')
            remapped_state_dict[mapped_key] = value
        self.model.load_state_dict(remapped_state_dict)

    def train(self, setting):
        train_loader, val_loader, test_loader, scaler = self._get_data()

        time_now = time.time()
        path = os.path.join(self.args.checkpoints,self.args.model, self.args.dataset ,self.args.condition,setting)
        if not os.path.exists(path):
            os.makedirs(path)
        diagnostics_path = os.path.join(
            getattr(self.args, 'results_dir', self.args.checkpoints),
            self.args.model,
            self.args.dataset,
            self.args.condition,
            setting,
        )
        os.makedirs(diagnostics_path, exist_ok=True)
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)


        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        self.validation_diagnostics_history = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_diversity_loss= []
            

            self.model.train()
            epoch_time = time.time()
            for inputs, targets in train_loader:
                capacity_increment, relaxation_features, charge_current, discharge_current,Temperature = inputs
                capacity_increment = capacity_increment.to(self.device)
                relaxation_features = relaxation_features.to(self.device)
                charge_current = charge_current.to(self.device)
                discharge_current = discharge_current.to(self.device)
                Temperature = Temperature.to(self.device)
                targets = targets.to(self.device)

                outputs, gates = self.model(capacity_increment, relaxation_features, charge_current, discharge_current,Temperature)

                if self.args.model in ('iMOE', 'iMOE_CSR', 'iMOE_HD', 'iMOE_SDR'):
                    importance = gates.sum(0)  
                    diversity_loss = cv_squared(importance) 
                else:
                    diversity_loss = torch.tensor(0.0, device=self.device) 

                main_loss = criterion(outputs, targets)

                total_loss = main_loss + self.args.diverloss * diversity_loss
                train_loss.append(total_loss.item())
                train_diversity_loss.append(diversity_loss.item())  

                model_optim.zero_grad()
                total_loss.backward()
                model_optim.step()
            train_loss = np.average(train_loss)
            train_diversity_loss = np.average(train_diversity_loss)
            vali_loss = self.vali(train_loader, val_loader, criterion)
            self._save_validation_diagnostics(
                diagnostics_path,
                epoch + 1,
                vali_loss,
            )
            test_loss = vali_loss
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} diversity Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, train_diversity_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        best_model_path = path + '/' + 'checkpoint.pth'
        self._load_checkpoint(best_model_path)
            

        return self.model

    def _save_validation_diagnostics(self, path, epoch, validation_loss):
        diagnostics = getattr(self, 'last_validation_diagnostics', None)
        if diagnostics is None:
            return

        record = {
            'epoch': epoch,
            'validation_loss': float(validation_loss),
            **diagnostics,
        }
        self.validation_diagnostics_history.append(record)
        diagnostics_path = os.path.join(
            path,
            'validation_routing_diagnostics.json',
        )
        with open(diagnostics_path, 'w', encoding='utf-8') as diagnostics_file:
            json.dump(
                self.validation_diagnostics_history,
                diagnostics_file,
                indent=2,
            )
    
    def vali(self, train_loader, val_loader, criterion):
        self.model.eval()  
        
        val_loss = [] 
        prediction_squared_error = 0.0
        prediction_element_count = 0
        prediction_mse_only = getattr(self.args, 'selection_metric', None) == 'prediction_mse'
        routing_diagnostic_batches = {}
        
        with torch.no_grad():  
            for i, (batch_x, batch_y) in enumerate(val_loader):
                capacity_increment, relaxation_features, charge_current, discharge_current,Temperature = batch_x
                capacity_increment = capacity_increment.to(self.device)
                relaxation_features = relaxation_features.to(self.device)
                charge_current = charge_current.to(self.device)
                discharge_current = discharge_current.to(self.device)
                Temperature = Temperature.to(self.device)
                targets = batch_y.to(self.device)
 
                outputs, gates = self.model(capacity_increment, relaxation_features, charge_current, discharge_current,Temperature)

                diagnostic_model = getattr(self.model, 'module', self.model)
                get_diagnostics = getattr(
                    diagnostic_model,
                    'get_routing_diagnostics',
                    None,
                )
                if get_diagnostics is not None:
                    batch_diagnostics = get_diagnostics()
                    for name, values in batch_diagnostics.items():
                        routing_diagnostic_batches.setdefault(name, []).append(
                            values.detach().cpu().reshape(-1)
                        )

                if prediction_mse_only:
                    prediction_squared_error += torch.sum((outputs - targets) ** 2).item()
                    prediction_element_count += targets.numel()
                    continue

                if self.args.model in ('iMOE', 'iMOE_CSR', 'iMOE_HD', 'iMOE_SDR'):
                    importance = gates.sum(0)  
                    diversity_loss = cv_squared(importance)  
                else:
                    diversity_loss = 0  #
 
                main_loss = criterion(outputs, targets)
                
                total_loss = main_loss + self.args.diverloss * diversity_loss
                val_loss.append(total_loss.item())  

        if prediction_mse_only:
            avg_val_loss = prediction_squared_error / prediction_element_count
        else:
            avg_val_loss = np.mean(val_loss)
        if routing_diagnostic_batches:
            self.last_validation_diagnostics = {}
            for name, batches in routing_diagnostic_batches.items():
                values = torch.cat(batches).float()
                self.last_validation_diagnostics[f'{name}_mean'] = (
                    values.mean().item()
                )
                self.last_validation_diagnostics[f'{name}_std'] = (
                    values.std(unbiased=False).item()
                )
        else:
            self.last_validation_diagnostics = None
        self.model.train()
        return avg_val_loss


    def test(self, setting, test=0):
        train_loader, val_loader, test_loader, scaler = self._get_data()
        if test:
            print('loading model')
            if not self.args.checkpoint_path:
                raise ValueError('--checkpoint_path is required when --is_training=0')
            self._load_checkpoint(self.args.checkpoint_path)
        self.model.eval()

        data_loader = test_loader
        total_rmse = 0
        total_mape = 0
        count = 0
        all_true_values = []
        all_pred_values = []
        all_weights = []
        explainability_rows = []
        metadata_offset = 0
        sample_metadata = getattr(data_loader, 'sample_metadata', None)
        if self.args.model == 'iMOE_SDR':
            if sample_metadata is None:
                raise ValueError(
                    'iMOE_SDR test loader is missing sample_metadata'
                )
            if len(sample_metadata) != len(data_loader.dataset):
                raise ValueError(
                    'sample_metadata length does not match test dataset length'
                )
        start_time = time.time()
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                capacity_increment, relaxation_features, charge_current, discharge_current, Temperature = X_batch
                capacity_increment = capacity_increment.to(self.device)
                relaxation_features = relaxation_features.to(self.device)
                charge_current = charge_current.to(self.device)
                discharge_current = discharge_current.to(self.device)
                Temperature = Temperature.to(self.device)
                targets = y_batch.to(self.device)

                outputs, gates = self.model(capacity_increment, relaxation_features, charge_current, discharge_current, Temperature)

                all_weights.append(gates.cpu().numpy())
                if self.args.model == 'iMOE_SDR':
                    batch_size = outputs.shape[0]
                    batch_metadata = sample_metadata[
                        metadata_offset:metadata_offset + batch_size
                    ]
                    diagnostic_model = getattr(self.model, 'module', self.model)
                    diagnostics = diagnostic_model.get_routing_diagnostics()
                    explainability_rows.extend(
                        self._build_sdr_explainability_rows(
                            gates,
                            diagnostics,
                            batch_metadata,
                        )
                    )
                    metadata_offset += batch_size

                if self.args.inverse == 'yes':  
                    y_pred_inv = scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).reshape(outputs.shape)
                    y_batch_inv = scaler.inverse_transform(targets.cpu().numpy().reshape(-1, 1)).reshape(targets.shape)
                else:
                    y_pred_inv = outputs.cpu().numpy().reshape(outputs.shape)
                    y_batch_inv = targets.cpu().numpy().reshape(targets.shape)

                eps = 1e-8  
                diff = y_pred_inv - y_batch_inv

                rmse = np.sqrt(np.nanmean(diff ** 2))
                denom = np.where(np.abs(y_batch_inv) < eps, np.nan, np.abs(y_batch_inv))
                mape = np.nanmean(np.abs(diff / denom)) * 100.0
                total_rmse += rmse
                total_mape += mape
                count += 1
                all_true_values.append(y_batch_inv)
                all_pred_values.append(y_pred_inv)
        if (
            self.args.model == 'iMOE_SDR'
            and metadata_offset != len(sample_metadata)
        ):
            raise ValueError(
                'sample_metadata was not consumed in test loader order'
            )
        end_time = time.time()
        total_test_time = end_time - start_time
        print(f"Total test time: {total_test_time:.2f} seconds")
        avg_rmse = total_rmse / count
        avg_mape = total_mape / count
        print(f"Average MAPE (Normalized): {avg_mape:.4f}%")
        print(f"Average Test RMSE: {avg_rmse:.4f}")
        all_true_values = np.concatenate(all_true_values, axis=0)
        all_pred_values = np.concatenate(all_pred_values, axis=0)
        all_weights = np.concatenate(all_weights, axis=0)
        results_root = getattr(self.args, 'results_dir', self.args.checkpoints)
        output_dir = os.path.join(
            results_root,
            self.args.model,
            self.args.dataset,
            self.args.condition,
            setting,
        )
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, 'true_values.npy'), all_true_values)
        np.save(os.path.join(output_dir, 'pred_values.npy'), all_pred_values)
        metrics_payload = {
            'prediction_shape': list(all_pred_values.shape),
            'global': _global_metrics(all_pred_values, all_true_values),
            'training_protocol': {
                'max_epochs': getattr(self.args, 'train_epochs', 1500),
                'batch_size': getattr(self.args, 'batch_size', 32),
                'patience': getattr(self.args, 'patience', 200),
                'learning_rate': getattr(self.args, 'learning_rate', None),
                'loss': getattr(self.args, 'loss', 'MSE'),
                'checkpoint_selection': 'validation_global_mse',
            },
            'evaluation_protocol': {
                'batch_size': getattr(
                    self.args,
                    'eval_batch_size',
                    getattr(self.args, 'batch_size', 32),
                ),
            },
            'data_protocol': {
                'data_root': (
                    portable_artifact_path(self.args.data_root)
                    if getattr(self.args, 'data_root', None)
                    else ''
                ),
                'requested_dataaccess_percent': getattr(
                    self.args,
                    'dataaccess',
                    100,
                ),
                'requested_train_battery_count': getattr(
                    self.args,
                    'train_battery_count',
                    None,
                ),
                'actual_train_battery_count': getattr(
                    self.args,
                    'actual_train_battery_count',
                    None,
                ),
                'available_train_batteries': getattr(
                    self.args,
                    'available_train_batteries',
                    None,
                ),
                'selected_train_batteries': getattr(
                    self.args,
                    'selected_train_batteries',
                    None,
                ),
            },
        }
        with open(
            os.path.join(output_dir, 'metrics.json'),
            'w',
            encoding='utf-8',
        ) as metrics_file:
            json.dump(metrics_payload, metrics_file, indent=2)
        if self.args.model in ('iMOE', 'iMOE_CSR', 'iMOE_HD', 'iMOE_SDR'):
            weights_df = pd.DataFrame(all_weights)
            weights_df.to_csv(os.path.join(output_dir, 'weights.csv'), index=False)
        if self.args.model == 'iMOE_SDR':
            pd.DataFrame(explainability_rows).to_csv(
                os.path.join(output_dir, 'sdr_explainability.csv'),
                index=False,
            )
        colors = ["#403990", "#80A6E2", "#FBBD85", "#F46F43", "#CF3D3E"]
        cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
        if self.args.model in ('iMOE', 'iMOE_CSR', 'iMOE_HD', 'iMOE_SDR'):
            plt.figure(figsize=(12, 6))
            sns.heatmap(all_weights.T, cmap=cmap, cbar=True, yticklabels=[f'Weight {i + 1}' for i in range(all_weights.shape[1])])
            plt.title('Weights for Each Sample')
            plt.xlabel('Sample Index')
            plt.ylabel('Weights')
            plt.savefig(os.path.join(output_dir, 'weights_heatmap.png')) 
            plt.close()  
        plt.figure(figsize=(15, 8))
        for i in range(len(all_true_values)):
            plt.plot(range(i, i + len(all_true_values[i])), all_true_values[i], color='blue', alpha=0.5, label='True Values' if i == 0 else "")
        for i in range(len(all_pred_values)):
            plt.plot(range(i, i + len(all_pred_values[i])), all_pred_values[i], color='red', alpha=0.5, label='Predictions' if i == 0 else "")
        plt.legend()
        plt.title('All Samples: True Values vs Predictions')
        plt.xlabel('Cycle Index')
        plt.ylabel('Discharge Capacity')
        plt.savefig(os.path.join(output_dir, 'true_vs_predicted.png')) 
        plt.close()  
        return avg_rmse, avg_mape

    @staticmethod
    def _build_sdr_explainability_rows(gates, diagnostics, metadata):
        required_diagnostics = (
            'fusion_gate',
            'router_weight_l1',
            'statistical_route_entropy',
            'curve_route_entropy',
            'fused_route_entropy',
        )
        batch_size = gates.shape[0]
        if len(metadata) != batch_size:
            raise ValueError(
                'sample_metadata batch does not match model output batch'
            )
        missing = [
            name for name in required_diagnostics if name not in diagnostics
        ]
        if missing:
            raise ValueError(
                f'Missing iMOE_SDR routing diagnostics: {missing}'
            )

        diagnostic_values = {}
        for name in required_diagnostics:
            values = diagnostics[name].detach().cpu().reshape(-1)
            if values.numel() != batch_size:
                raise ValueError(
                    f'{name} length does not match model output batch'
                )
            diagnostic_values[name] = values.tolist()

        gate_values = gates.detach().cpu()
        rows = []
        for sample_index, sample_metadata in enumerate(metadata):
            row = dict(sample_metadata)
            for name in required_diagnostics:
                row[name] = diagnostic_values[name][sample_index]
            for expert_index, weight in enumerate(gate_values[sample_index]):
                row[f'expert_{expert_index + 1}_fused_weight'] = weight.item()
            rows.append(row)
        return rows
