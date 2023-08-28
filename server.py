import argparse
import itertools
from typing import List, Tuple, Optional, Dict
from functools import reduce
import flwr as fl
import numpy as np
from flwr.common import Metrics
from typing import Callable, Union
import operator
from omegaconf import OmegaConf
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from Moudules.Lightning_MHBAMixer_Module import MHBAMixerModule

configs = OmegaConf.load("config.yml")
dataset_conf, model_conf = configs.dataset, configs.model
num_model1, num_model2, num_model3 = model_conf.num_models
model1_layers, model2_layers, model3_layers = 0, 0, 0
def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    results = itertools.groupby(results, key=lambda i: i[0][0])
    weights_prime_list = {}
    for k, result in results:
        result = list(result)
        if result == []:
            continue
        num_examples_total = sum([num_examples for _, num_examples in result])
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in result
        ]
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        if weights_prime[0] == model_conf.model_flag[0]:
            weights_prime_list["model1"] = weights_prime
        elif weights_prime[0] == model_conf.model_flag[1]:
            weights_prime_list["model2"] = weights_prime
        else:
            weights_prime_list["model3"] = weights_prime
    weights_prime_list = list(weights_prime_list.values())
    weights_prime_list = sorted(weights_prime_list, key= lambda i: i[0])
    weights_prime = list(itertools.chain.from_iterable(weights_prime_list))
    return weights_prime


class FedCustom(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 6,
        min_evaluate_clients: int = 6,
        min_available_clients: int = 6,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        global model1_layers
        global model2_layers
        global model3_layers
        # vocab_size, n_heads, max_seq_len: int, hidden_dim: int, index: int, kernel_size: int,
        #                  dilation: int, padding: int, num_mixers: int, num_classes:
        model1 = MHBAMixerModule(model_flag=model_conf.model_flag[0],
                                 vocab_size=model_conf.vocab_size,
                                 n_heads=model_conf.n_heads,
                                 max_seq_len=model_conf.max_seq_len,
                                 hidden_dim=model_conf.hidden_dim,
                                 index=model_conf.index,
                                 kernel_size=model_conf.kernel_size,
                                 dilation=model_conf.dilation,
                                 padding=model_conf.padding,
                                 num_mixers=model_conf.num_mixers,
                                 num_classes=model_conf.num_classes,
                                 model_name="MHBAMixer")
        model2 = MHBAMixerModule(model_flag=model_conf.model_flag[1],
                                 vocab_size=model_conf.vocab_size,
                                 n_heads=model_conf.n_heads,
                                 max_seq_len=model_conf.max_seq_len,
                                 hidden_dim=model_conf.hidden_dim,
                                 index=model_conf.index,
                                 kernel_size=model_conf.kernel_size,
                                 dilation=model_conf.dilation,
                                 padding=model_conf.padding,
                                 num_mixers=model_conf.num_mixers,
                                 num_classes=model_conf.num_classes,
                                 model_name="DWTMixer")
        model3 = MHBAMixerModule(model_flag=model_conf.model_flag[2],
                                 vocab_size=model_conf.vocab_size,
                                 n_heads=model_conf.n_heads,
                                 max_seq_len=model_conf.max_seq_len,
                                 hidden_dim=model_conf.hidden_dim,
                                 index=model_conf.index,
                                 kernel_size=model_conf.kernel_size,
                                 dilation=model_conf.dilation,
                                 padding=model_conf.padding,
                                 num_mixers=model_conf.num_mixers,
                                 num_classes=model_conf.num_classes,
                                 model_name="TSMixer")
        ndarrays_1 = _get_parameters(model1)
        ndarrays_2 = _get_parameters(model2)
        ndarrays_3 = _get_parameters(model3)
        merge_ndarrays = ndarrays_1+ndarrays_2+ndarrays_3
        model1_layers = len(model1.state_dict().keys())
        model2_layers = len(model2.state_dict().keys())
        model3_layers = len(model3.state_dict().keys())
        return fl.common.ndarrays_to_parameters(merge_ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        higher_lr_config = {"lr": 0.003}
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, higher_lr_config))
                )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        agg_res = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(agg_res)
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1score = [num_examples * m["f1score"] for num_examples, m in metrics]
    recall = [num_examples * m["recall"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples),
                "f1score": sum(f1score)/sum(examples),
                "recall": sum(recall)/sum(examples)}

def main() -> None:
    # Define strategy
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=10)
    args = parser.parse_args()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.max_epoch),
        strategy=FedCustom(evaluate_metrics_aggregation_fn=weighted_average),
    )


if __name__ == "__main__":
    main()
