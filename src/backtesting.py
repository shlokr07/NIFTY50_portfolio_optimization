# backtesting.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
from tqdm import tqdm
import warnings
from datetime import datetime, timedelta

# Import model from training file
from model_training import QuantileTransformer, CONFIG as TRAINING_CONFIG

warnings.filterwarnings('ignore')

class QuantileBacktester:
    """
    Backtesting framework for quantile-based portfolio strategies
    """
    
    def __init__(self, model_path: str, data_dir: str, device: str = None):
        """
        Initialize backtester with trained model and data
        
        Args:
            model_path: Path to saved model
            data_dir: Directory containing processed data
            device: Device to run model on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.data_dir = data_dir
        
        # Load model and data
        self.model, self.model_info = self._load_model()
        self.data = self._load_data()
        
        # Extract key information
        self.tickers = self.data['tickers']
        self.features = self.data['features']
        self.quantiles = self.model_info['config']['QUANTILES']
        self.horizons = self.model_info['model_info']['data_config']['horizons']
        
        print(f"Loaded model with {len(self.tickers)} stocks")
        print(f"Quantiles: {self.quantiles}")
        print(f"Horizons: {self.horizons}")
        
    def _load_model(self):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_arch = checkpoint['model_architecture']
        
        # Recreate model
        model = QuantileTransformer(
            seq_len=model_arch['seq_len'],
            num_stocks=model_arch['num_stocks'],
            num_features=model_arch['num_features'],
            patch_size=model_arch['patch_size'],
            d_model=model_arch['d_model'],
            n_heads=model_arch['n_heads'],
            num_layers=model_arch['num_layers'],
            dropout=model_arch['dropout'],
            quantiles=model_arch['quantiles'],
            num_horizons=model_arch['num_horizons']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, checkpoint
        
    def _load_data(self):
        """Load all data splits and metadata"""
        data = {}
        
        # Load arrays
        for split in ['train', 'val', 'test', 'full']:
            data[split] = {}
            for data_type in ['X', 'Y', 'mask', 'dates']:
                file_path = f"{self.data_dir}/{split}_{data_type}.npy"
                if os.path.exists(file_path):
                    data[split][data_type] = np.load(file_path, allow_pickle=True)
        
        # Load metadata
        data['tickers'] = np.load(f"{self.data_dir}/tickers.npy", allow_pickle=True)
        data['features'] = np.load(f"{self.data_dir}/features.npy", allow_pickle=True)
        
        with open(f"{self.data_dir}/config.json", 'r') as f:
            data['config'] = json.load(f)
            
        return data
        
    def generate_predictions(self, data_split: str = 'test') -> Dict:
        """
        Generate predictions for specified data split
        
        Args:
            data_split: Which data split to use ('train', 'val', 'test', 'full')
            
        Returns:
            Dictionary containing predictions and metadata
        """
        print(f"Generating predictions for {data_split} split...")
        
        X = torch.tensor(self.data[data_split]['X'], dtype=torch.float32).to(self.device)
        Y = self.data[data_split]['Y']
        mask = self.data[data_split]['mask']
        dates = self.data[data_split]['dates']
        
        predictions = []
        batch_size = 32  # Process in batches to avoid memory issues
        
        with torch.no_grad():
            for i in tqdm(range(0, len(X), batch_size), desc="Predicting"):
                batch_X = X[i:i+batch_size]
                batch_pred = self.model(batch_X)  # [B, S, Q, H]
                predictions.append(batch_pred.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        return {
            'predictions': predictions,  # [T, S, Q, H]
            'actuals': Y,               # [T, S, H]
            'mask': mask,               # [T, S]
            'dates': dates,             # [T]
            'tickers': self.tickers,
            'quantiles': self.quantiles,
            'horizons': self.horizons
        }
    
    def create_portfolio_strategy(self, predictions_dict: Dict, 
                                top_k: int = 5, 
                                horizon_idx: int = 0,
                                risk_quantile_idx: int = 0) -> pd.DataFrame:
        """
        Create portfolio strategy based on quantile predictions
        
        Args:
            predictions_dict: Output from generate_predictions
            top_k: Number of top stocks to select
            horizon_idx: Which horizon to use (0 for first horizon)
            risk_quantile_idx: Which quantile to use for risk assessment (0=0.1, 1=0.5, 2=0.9)
            
        Returns:
            DataFrame with portfolio weights over time
        """
        predictions = predictions_dict['predictions']  # [T, S, Q, H]
        dates = predictions_dict['dates']
        tickers = predictions_dict['tickers']
        mask = predictions_dict['mask']
        
        # Get median quantile predictions (usually index 1 for [0.1, 0.5, 0.9])
        median_idx = len(self.quantiles) // 2
        median_preds = predictions[:, :, median_idx, horizon_idx]  # [T, S]
        
        # Get risk quantile predictions
        risk_preds = predictions[:, :, risk_quantile_idx, horizon_idx]  # [T, S]
        
        portfolio_weights = []
        
        for t in range(len(dates)):
            # Get valid stocks for this time step
            valid_mask = mask[t]  # [S]
            
            if not np.any(valid_mask):
                # No valid stocks, equal weight
                weights = np.ones(len(tickers)) / len(tickers)
            else:
                # Get predictions for valid stocks
                valid_median = median_preds[t][valid_mask]
                valid_risk = risk_preds[t][valid_mask]
                valid_tickers = tickers[valid_mask]
                
                # Simple strategy: Select top K stocks by median prediction
                # with risk adjustment (avoid stocks with very negative risk predictions)
                risk_threshold = np.percentile(valid_risk, 25)  # Bottom 25% risk threshold
                
                # Filter out high-risk stocks
                low_risk_mask = valid_risk >= risk_threshold
                if np.any(low_risk_mask):
                    filtered_median = valid_median[low_risk_mask]
                    filtered_tickers_idx = np.where(valid_mask)[0][low_risk_mask]
                else:
                    # If all stocks are high risk, use all valid stocks
                    filtered_median = valid_median
                    filtered_tickers_idx = np.where(valid_mask)[0]
                
                # Select top K stocks
                if len(filtered_median) >= top_k:
                    top_indices = np.argsort(filtered_median)[-top_k:]
                    selected_stocks = filtered_tickers_idx[top_indices]
                else:
                    # If fewer than top_k stocks available, use all filtered stocks
                    selected_stocks = filtered_tickers_idx
                
                # Create weights (equal weight among selected stocks)
                weights = np.zeros(len(tickers))
                if len(selected_stocks) > 0:
                    weights[selected_stocks] = 1.0 / len(selected_stocks)
            
            portfolio_weights.append(weights)
        
        # Create DataFrame
        weights_df = pd.DataFrame(
            portfolio_weights, 
            index=dates, 
            columns=tickers
        )
        
        return weights_df
    
    def calculate_portfolio_returns(self, weights_df: pd.DataFrame, 
                                  predictions_dict: Dict, 
                                  horizon_idx: int = 0) -> pd.DataFrame:
        """
        Calculate portfolio returns based on weights and actual returns
        
        Args:
            weights_df: Portfolio weights over time
            predictions_dict: Contains actual returns
            horizon_idx: Which horizon to use for returns
            
        Returns:
            DataFrame with portfolio returns
        """
        actuals = predictions_dict['actuals'][:, :, horizon_idx]  # [T, S]
        dates = predictions_dict['dates']
        tickers = predictions_dict['tickers']
        mask = predictions_dict['mask']
        
        # Calculate portfolio returns
        portfolio_returns = []
        individual_returns = []
        
        for t in range(len(dates)):
            weights = weights_df.iloc[t].values
            returns = actuals[t]  # Individual stock returns (log returns)
            valid_mask = mask[t]
            
            # Only consider valid stocks
            valid_weights = weights[valid_mask]
            valid_returns = returns[valid_mask]
            
            if len(valid_weights) > 0 and np.sum(valid_weights) > 0:
                # Normalize weights
                valid_weights = valid_weights / np.sum(valid_weights)
                
                # Portfolio return (weighted average of log returns)
                portfolio_return = np.sum(valid_weights * valid_returns)
            else:
                portfolio_return = 0.0
            
            portfolio_returns.append(portfolio_return)
            individual_returns.append(returns)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': dates,
            'portfolio_return': portfolio_returns,
            'cumulative_return': np.cumsum(portfolio_returns),
            'portfolio_value': np.exp(np.cumsum(portfolio_returns))  # Convert log returns to price
        })
        
        # Add individual stock returns for analysis
        stock_returns_df = pd.DataFrame(
            individual_returns, 
            index=dates, 
            columns=tickers
        )
        
        return results_df, stock_returns_df
    
    def calculate_benchmark_returns(self, predictions_dict: Dict, 
                                  horizon_idx: int = 0) -> pd.DataFrame:
        """
        Calculate equal-weighted benchmark returns
        
        Args:
            predictions_dict: Contains actual returns
            horizon_idx: Which horizon to use
            
        Returns:
            DataFrame with benchmark returns
        """
        actuals = predictions_dict['actuals'][:, :, horizon_idx]  # [T, S]
        dates = predictions_dict['dates']
        mask = predictions_dict['mask']
        
        benchmark_returns = []
        
        for t in range(len(dates)):
            returns = actuals[t]
            valid_mask = mask[t]
            
            if np.any(valid_mask):
                # Equal weight benchmark
                valid_returns = returns[valid_mask]
                benchmark_return = np.mean(valid_returns)
            else:
                benchmark_return = 0.0
            
            benchmark_returns.append(benchmark_return)
        
        benchmark_df = pd.DataFrame({
            'date': dates,
            'benchmark_return': benchmark_returns,
            'benchmark_cumulative': np.cumsum(benchmark_returns),
            'benchmark_value': np.exp(np.cumsum(benchmark_returns))
        })
        
        return benchmark_df
    
    def run_backtest(self, data_split: str = 'test', 
                    top_k: int = 5, 
                    horizon_idx: int = 0,
                    risk_quantile_idx: int = 0) -> Dict:
        """
        Run complete backtest
        
        Args:
            data_split: Which data split to use
            top_k: Number of stocks to select
            horizon_idx: Which horizon to use
            risk_quantile_idx: Which quantile for risk assessment
            
        Returns:
            Dictionary with all backtest results
        """
        print(f"Running backtest...")
        print(f"Strategy: Top {top_k} stocks, Horizon: {self.horizons[horizon_idx]} days")
        
        # Generate predictions
        predictions_dict = self.generate_predictions(data_split)
        
        # Create portfolio strategy
        weights_df = self.create_portfolio_strategy(
            predictions_dict, top_k, horizon_idx, risk_quantile_idx
        )
        
        # Calculate returns
        portfolio_df, stock_returns_df = self.calculate_portfolio_returns(
            weights_df, predictions_dict, horizon_idx
        )
        
        # Calculate benchmark
        benchmark_df = self.calculate_benchmark_returns(predictions_dict, horizon_idx)
        
        # Combine results
        results_df = portfolio_df.merge(benchmark_df, on='date')
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(results_df)
        
        return {
            'results': results_df,
            'weights': weights_df,
            'stock_returns': stock_returns_df,
            'performance': performance_metrics,
            'predictions': predictions_dict,
            'config': {
                'top_k': top_k,
                'horizon_idx': horizon_idx,
                'horizon_days': self.horizons[horizon_idx],
                'risk_quantile_idx': risk_quantile_idx,
                'data_split': data_split
            }
        }
    
    def _calculate_performance_metrics(self, results_df: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        portfolio_returns = results_df['portfolio_return'].values
        benchmark_returns = results_df['benchmark_return'].values
        
        # Basic metrics
        total_return = results_df['portfolio_value'].iloc[-1] - 1
        benchmark_total_return = results_df['benchmark_value'].iloc[-1] - 1
        
        # Annualized metrics (assuming daily returns)
        trading_days = 252
        n_days = len(portfolio_returns)
        years = n_days / trading_days
        
        annualized_return = (results_df['portfolio_value'].iloc[-1] ** (1/years)) - 1
        benchmark_annualized_return = (results_df['benchmark_value'].iloc[-1] ** (1/years)) - 1
        
        # Volatility
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(trading_days)
        benchmark_vol = np.std(benchmark_returns) * np.sqrt(trading_days)
        
        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe_ratio = annualized_return / portfolio_vol if portfolio_vol > 0 else 0
        benchmark_sharpe = benchmark_annualized_return / benchmark_vol if benchmark_vol > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = results_df['cumulative_return'].values
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown)
        
        # Excess returns
        excess_returns = portfolio_returns - benchmark_returns
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(trading_days) if np.std(excess_returns) > 0 else 0
        
        # Win rate
        win_rate = np.mean(portfolio_returns > 0)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'information_ratio': information_ratio,
            'benchmark_total_return': benchmark_total_return,
            'benchmark_annualized_return': benchmark_annualized_return,
            'benchmark_volatility': benchmark_vol,
            'benchmark_sharpe_ratio': benchmark_sharpe,
            'excess_return': total_return - benchmark_total_return,
            'trading_days': n_days,
            'years': years
        }
    
    def plot_results(self, backtest_results: Dict, save_path: str = None):
        """
        Plot backtest results
        
        Args:
            backtest_results: Results from run_backtest
            save_path: Path to save plots
        """
        results_df = backtest_results['results']
        weights_df = backtest_results['weights']
        performance = backtest_results['performance']
        config = backtest_results['config']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Portfolio Backtest Results - Top {config["top_k"]} Strategy', fontsize=16)
        
        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        ax1.plot(results_df['date'], results_df['portfolio_value'], label='Portfolio', linewidth=2)
        ax1.plot(results_df['date'], results_df['benchmark_value'], label='Equal Weight Benchmark', linewidth=2)
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Daily Returns
        ax2 = axes[0, 1]
        ax2.plot(results_df['date'], results_df['portfolio_return'], alpha=0.7, label='Portfolio')
        ax2.plot(results_df['date'], results_df['benchmark_return'], alpha=0.7, label='Benchmark')
        ax2.set_title('Daily Returns')
        ax2.set_ylabel('Daily Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Portfolio Weights Heatmap
        ax3 = axes[1, 0]
        # Sample weights for visualization (show last 60 days)
        sample_weights = weights_df.iloc[-60:] if len(weights_df) > 60 else weights_df
        # Only show stocks that were selected at least once
        selected_stocks = sample_weights.columns[sample_weights.sum() > 0]
        sample_weights_filtered = sample_weights[selected_stocks]
        
        if len(sample_weights_filtered.columns) > 0:
            sns.heatmap(sample_weights_filtered.T, ax=ax3, cmap='YlOrRd', 
                       cbar_kws={'label': 'Weight'}, xticklabels=False)
            ax3.set_title('Portfolio Weights (Last 60 Days)')
            ax3.set_ylabel('Stocks')
        else:
            ax3.text(0.5, 0.5, 'No weights to display', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Portfolio Weights')
        
        # 4. Performance Metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics_text = f"""
        Performance Metrics:
        
        Total Return: {performance['total_return']:.2%}
        Annualized Return: {performance['annualized_return']:.2%}
        Volatility: {performance['volatility']:.2%}
        Sharpe Ratio: {performance['sharpe_ratio']:.3f}
        Max Drawdown: {performance['max_drawdown']:.2%}
        Win Rate: {performance['win_rate']:.2%}
        
        Benchmark Comparison:
        Benchmark Return: {performance['benchmark_total_return']:.2%}
        Excess Return: {performance['excess_return']:.2%}
        Information Ratio: {performance['information_ratio']:.3f}
        
        Period: {performance['trading_days']} days ({performance['years']:.1f} years)
        """
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def analyze_predictions(self, predictions_dict: Dict, save_path: str = None):
        """
        Analyze prediction quality
        
        Args:
            predictions_dict: Predictions from generate_predictions
            save_path: Path to save analysis plots
        """
        predictions = predictions_dict['predictions']  # [T, S, Q, H]
        actuals = predictions_dict['actuals']  # [T, S, H]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prediction Quality Analysis', fontsize=16)
        
        # For each horizon
        for h_idx, horizon in enumerate(self.horizons):
            if h_idx >= 2:  # Only plot first 2 horizons
                break
                
            pred_h = predictions[:, :, :, h_idx]  # [T, S, Q]
            actual_h = actuals[:, :, h_idx]  # [T, S]
            
            # Flatten for analysis
            pred_flat = pred_h.reshape(-1, len(self.quantiles))
            actual_flat = actual_h.reshape(-1)
            
            # Remove NaN values
            valid_mask = ~np.isnan(actual_flat)
            pred_flat = pred_flat[valid_mask]
            actual_flat = actual_flat[valid_mask]
            
            # 1. Prediction vs Actual scatter
            ax1 = axes[h_idx, 0]
            median_idx = len(self.quantiles) // 2
            ax1.scatter(actual_flat, pred_flat[:, median_idx], alpha=0.5, s=1)
            ax1.plot([actual_flat.min(), actual_flat.max()], [actual_flat.min(), actual_flat.max()], 
                    'r--', label='Perfect Prediction')
            ax1.set_xlabel('Actual Returns')
            ax1.set_ylabel('Predicted Returns (Median)')
            ax1.set_title(f'Predictions vs Actuals - {horizon}d Horizon')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Quantile coverage
            ax2 = axes[h_idx, 1]
            coverage = []
            for q_idx, quantile in enumerate(self.quantiles):
                if quantile < 0.5:
                    # Lower quantile: actual should be above prediction
                    coverage.append(np.mean(actual_flat >= pred_flat[:, q_idx]))
                else:
                    # Upper quantile: actual should be below prediction
                    coverage.append(np.mean(actual_flat <= pred_flat[:, q_idx]))
            
            ax2.bar(range(len(self.quantiles)), coverage, alpha=0.7)
            ax2.axhline(y=1.0, color='r', linestyle='--', label='Perfect Coverage')
            ax2.set_xlabel('Quantiles')
            ax2.set_ylabel('Coverage Rate')
            ax2.set_title(f'Quantile Coverage - {horizon}d Horizon')
            ax2.set_xticks(range(len(self.quantiles)))
            ax2.set_xticklabels([f'{q:.1f}' for q in self.quantiles])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Analysis plot saved to: {save_path}")
        
        plt.show()


def main():
    """
    Main function to run backtesting
    """
    # Configuration
    MODEL_PATH = TRAINING_CONFIG["MODEL_PATH"]
    DATA_DIR = TRAINING_CONFIG["DATA_DIR"]
    RESULTS_DIR = "results/backtesting"
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*80)
    print("QUANTILE PORTFOLIO BACKTESTING")
    print("="*80)
    
    # Initialize backtester
    backtester = QuantileBacktester(MODEL_PATH, DATA_DIR)
    
    # Run backtest with different configurations
    strategies = [
        {'top_k': 5, 'horizon_idx': 0, 'risk_quantile_idx': 0, 'name': 'Top5_H20_RiskQ10'},
        {'top_k': 5, 'horizon_idx': 1, 'risk_quantile_idx': 0, 'name': 'Top5_H60_RiskQ10'},
        {'top_k': 10, 'horizon_idx': 0, 'risk_quantile_idx': 1, 'name': 'Top10_H20_RiskQ50'},
    ]
    
    all_results = {}
    
    for strategy in strategies:
        print(f"\nRunning strategy: {strategy['name']}")
        print("-" * 40)
        
        # Run backtest
        results = backtester.run_backtest(
            data_split='test',
            top_k=strategy['top_k'],
            horizon_idx=strategy['horizon_idx'],
            risk_quantile_idx=strategy['risk_quantile_idx']
        )
        
        all_results[strategy['name']] = results
        
        # Plot results
        plot_path = f"{RESULTS_DIR}/{strategy['name']}_backtest.png"
        backtester.plot_results(results, plot_path)
        
        # Print performance summary
        perf = results['performance']
        print(f"\nPerformance Summary for {strategy['name']}:")
        print(f"  Total Return: {perf['total_return']:.2%}")
        print(f"  Annualized Return: {perf['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {perf['max_drawdown']:.2%}")
        print(f"  Excess Return vs Benchmark: {perf['excess_return']:.2%}")
    
    # Analyze predictions
    print("\nAnalyzing prediction quality...")
    predictions_dict = backtester.generate_predictions('test')
    analysis_path = f"{RESULTS_DIR}/prediction_analysis.png"
    backtester.analyze_predictions(predictions_dict, analysis_path)
    
    # Compare strategies
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        name: {
            'Total Return': f"{results['performance']['total_return']:.2%}",
            'Annualized Return': f"{results['performance']['annualized_return']:.2%}",
            'Sharpe Ratio': f"{results['performance']['sharpe_ratio']:.3f}",
            'Max Drawdown': f"{results['performance']['max_drawdown']:.2%}",
            'Excess Return': f"{results['performance']['excess_return']:.2%}",
            'Win Rate': f"{results['performance']['win_rate']:.2%}"
        }
        for name, results in all_results.items()
    })
    
    print(comparison_df.T)
    
    # Save results summary
    summary_path = f"{RESULTS_DIR}/backtest_summary.json"
    summary = {
        'strategies': {
            name: {
                'config': results['config'],
                'performance': results['performance']
            }
            for name, results in all_results.items()
        },
        'model_info': backtester.model_info['model_info'],
        'data_info': {
            'num_stocks': len(backtester.tickers),
            'features': backtester.features.tolist(),
            'quantiles': backtester.quantiles,
            'horizons': backtester.horizons
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nBacktest summary saved to: {summary_path}")
    print("Backtesting completed successfully!")


if __name__ == "__main__":
    main()