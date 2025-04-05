import os
import argparse
from hetu.utils.parallel import read_strategy, generate_ds

def process_strategies(strategy_path, model_name):
    for filename in os.listdir(strategy_path):
        if filename.endswith(".json"):
            json_path = os.path.join(strategy_path, filename)
            strategy_config = read_strategy(json_path)
            generate_ds(strategy_config, model_name)
            print(f"Successfully processed: {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate distributed strategies for model")
    parser.add_argument(
        "-p", "--strategy_path",
        required=True,
        help="Path to directory containing strategy JSON files"
    )
    parser.add_argument(
        "-m", "--model_name",
        required=True,
        help="Name of the target model"
    )
    
    args = parser.parse_args()
    
    # 验证路径有效性
    if not os.path.exists(args.strategy_path):
        raise FileNotFoundError(f"Strategy path not found: {args.strategy_path}")
    if not os.path.isdir(args.strategy_path):
        raise NotADirectoryError(f"Not a directory: {args.strategy_path}")

    process_strategies(args.strategy_path, args.model_name)
    print("All strategies ds generated")

if __name__ == "__main__":
    main()