# EEGMamba デバッグ機能 / Debug Functionality

このドキュメントはEEGMambaプロジェクトに実装されたデバッグ機能について説明します。  
This document explains the debugging functionality implemented in the EEGMamba project.

## 主な機能 / Main Features

### 1. 包括的なテンソル監視 / Comprehensive Tensor Monitoring
- テンソルの形状、統計情報（平均、標準偏差、ノルム）を自動ログ
- NaN/Inf値の検出と警告
- デバッグ用テンソルの保存機能

### 2. 勾配解析 / Gradient Analysis
- 勾配ノルムの監視
- 勾配消失/爆発の検出
- レイヤー別勾配統計

### 3. パフォーマンス分析 / Performance Analysis
- 関数実行時間の測定
- GPU メモリ使用量の監視
- 専門家（Expert）利用率の分析（MoE用）

### 4. カラー出力 / Colored Output
- 重要度に応じたカラー表示
- エラー・警告の視覚的強調

## 使用方法 / Usage

### 基本的な使用 / Basic Usage

```python
from debug_utils import enable_debug, debug_print, debug_tensor

# デバッグ機能を有効化
enable_debug()

# デバッグメッセージの出力
debug_print("This is a debug message", 'green')

# テンソルの監視
x = torch.randn(4, 128)
debug_tensor(x, "input_tensor")
```

### 設定例 / Configuration Examples

```python
from debug_config import get_debug_config

# 基本設定
basic_config = get_debug_config('basic')

# 高度な設定（テンソル保存含む）
advanced_config = get_debug_config('advanced')

# 本番環境設定（最小限のデバッグ）
production_config = get_debug_config('production')
```

### 訓練での使用 / Usage in Training

```python
# train.py内で
from debug_utils import enable_debug, check_gradients, debug_summary

# 訓練開始時にデバッグを有効化
enable_debug(save_tensors=False)

# 勾配チェック（10ステップごと）
if epoch % 10 == 0:
    grad_norms = check_gradients(model)

# 訓練終了時にサマリー表示
debug_summary()
```

## ファイル構成 / File Structure

- `debug_utils.py`: メインのデバッグユーティリティ
- `debug_config.py`: デバッグ設定の管理
- `test_debug.py`: デバッグ機能のテストスクリプト
- `DEBUG_README.md`: このドキュメント

## 追加された機能 / Added Features

### 1. EEGMamba モデル (`eegMamba.py`)
- フォワードパス全体の監視
- 各ブロックの出力チェック
- NaN/Inf検出

### 2. BidirectionalMamba (`bi_mammba.py`)
- 順方向/逆方向処理の監視
- 残差接続の検証
- タイミング分析

### 3. MoE (Mixture of Experts) (`moe.py`)
- 専門家利用率の分析
- ゲーティング重みの監視
- 出力統計の追跡

### 4. 訓練スクリプト (`train.py`)
- エポック別進捗監視
- 勾配フロー分析
- パフォーマンス統計

## デバッグ出力例 / Debug Output Examples

```
[DEBUG] === EEGMamba Forward Pass ===
[DEBUG] input: shape=(4, 62, 1000), mean=0.0123, std=0.9876, norm=124.56
[DEBUG] ST Adaptive output shape: torch.Size([4, 129, 512])
[DEBUG] st_adaptive_output: shape=(4, 129, 512), mean=0.0456, std=0.2134, norm=89.23
[DEBUG] Expert utilization: [0.8, 0.6, 0.9, 0.4, 0.7, 0.5, 0.8, 0.3]
[DEBUG] bidirectional_mamba_forward took 0.0234s
[DEBUG] Final output shape: torch.Size([4, 1024])
```

## テスト方法 / Testing

デバッグ機能をテストするには:
```bash
python test_debug.py
```

## 注意事項 / Notes

1. **パフォーマンスへの影響**: デバッグ機能は計算コストを増加させます。本番環境では`debug_mode=False`に設定してください。

2. **メモリ使用量**: `save_debug_tensors=True`の場合、大量のディスク容量が必要になる可能性があります。

3. **GPU メモリ**: CUDA使用時のみGPUメモリ監視が有効です。

## トラブルシューティング / Troubleshooting

### よくある問題 / Common Issues

1. **ImportError**: `termcolor`パッケージが不足している場合
   ```bash
   pip install termcolor
   ```

2. **CUDA out of memory**: デバッグ機能がメモリ使用量を増加させる場合
   - バッチサイズを削減
   - `save_debug_tensors=False`に設定

3. **ログが表示されない**: `debug_mode=False`になっている可能性
   ```python
   enable_debug()  # デバッグを有効化
   ```

## カスタマイズ / Customization

独自のデバッグ機能を追加するには:

```python
from debug_utils import debugger

# カスタムデバッグ関数
@debugger.timing_decorator("custom_function")
def my_custom_function(x):
    debugger.debug_print("Custom debug message")
    debugger.log_tensor_stats(x, "custom_tensor")
    return x * 2
```

このデバッグシステムにより、EEGMambaモデルの動作を詳細に分析し、問題を迅速に特定することが可能になります。