"""Optimization and forecasting without downloads or presentation generators."""
import argparse
from .experiment import run_experiment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/smoke.yaml')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--instances', nargs='+', default=None)
    parser.add_argument('--methods', nargs='+', choices=['exact','nsga2','nsga2_qp','hnsga2'])
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--forecast', action='store_true', help='Fit and select price models; do not optimize')
    args = parser.parse_args()
    if args.forecast:
        from .forecast_pipeline import build_forecasts
        print(build_forecasts(args.config))
        return
    path = run_experiment(args.config, run_id=args.run_id, continue_on_error=False,
        instances_override=args.instances, methods_override=args.methods,
        seeds_override=args.seeds)
    print(path)

if __name__ == '__main__':
    main()
