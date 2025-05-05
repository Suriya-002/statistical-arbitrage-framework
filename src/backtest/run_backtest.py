import argparse, yaml, pandas as pd, logging
from .engine import PairsBacktester

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', default='data/prices.csv')
    parser.add_argument('--start', default=None)
    parser.add_argument('--end', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    prices = pd.read_csv(args.data, index_col=0, parse_dates=True)
    results = PairsBacktester(config).run(prices, args.start, args.end)

    print("\n" + "="*60 + "\nBACKTEST RESULTS\n" + "="*60)
    for k, v in results['metrics'].items():
        if isinstance(v, float):
            fmt = f"{v:>10.2%}" if any(x in k for x in ['return','drawdown','win_rate']) else f"{v:>10.4f}"
            print(f"  {k:30s}: {fmt}")
    print("="*60)

if __name__ == '__main__':
    main()
