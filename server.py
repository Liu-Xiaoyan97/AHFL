import argparse
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


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
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        evaluate_metrics_aggregation_fn=weighted_average,
        # fit_metrics_aggregation_fn=weighted_average
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.max_epoch),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
