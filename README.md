一阶逻辑归结证明器 - 运行说明
====================================

本项目包含一个用于一阶逻辑的归结定理证明器，以及两个经典问题的形式化与求解实现。

## 文件结构

- `resolver.py`: 原始的基础版归结证明器。
- `resolver_improved.py`: 改进版证明器，增加了以下核心功能：
  1.  **变量标准化 (Standardize-Apart)**：在归结前对变量进行重命名，避免命名冲突，保证逻辑健全性。
  2.  **依赖追踪**：记录每个派生子句的来源，在导出空子句时可回溯生成完整的证明链。
  3.  **分离日志输出**：将详细的归结步骤输出到日志文件，控制台仅显示最终摘要。
- `trace_problem1.txt`: 运行改进版证明器后，针对 "Howling Hounds" 问题的详细轨迹日志。
- `trace_problem2.txt`: 运行改进版证明器后，针对 "Drug dealer and customs official" 问题的详细轨迹日志。
- `HowlingHounds_说明.md`: 对问题一的形式化与证明思路的说明。
- `DrugDealer_Customs_说明.md`: 对问题二的形式化与证明思路的说明。
- `Lab1_Resolution Theorem Prover.md`: 原始的实验要求文档。

## 运行环境

- **Python 3.6+**
- 无需安装任何第三方库，仅使用 Python 标准库。

## 运行方法

推荐运行改进版的证明器以获得更可靠且易于分析的结果。

1.  打开终端或命令行工具。
2.  切换到项目根目录（包含 `resolver_improved.py` 的目录）。
3.  执行以下命令：

    ```bash
    python resolver_improved.py
    ```

### 预期输出

-   **控制台输出**:
    程序将在控制台打印两行摘要信息，分别对应两个问题的证明结果，例如：

    ```
    Problem 1 => SUCCESS | steps: 16; used negated goal | log: trace_problem1.txt
    Problem 2 => SUCCESS | steps: 11; used negated goal | log: trace_problem2.txt
    ```
    这表示两个问题都成功找到了反证，并给出了归结步数与日志文件路径。

-   **文件输出**:
    程序会生成（或覆盖）两个日志文件：`trace_problem1.txt` 和 `trace_problem2.txt`。
    这些文件包含了详细的运算内容和结果，包括：
    
    -   初始的子句集（CNF）。
    -   每一步归结所用的子句、文字、以及发生的替换（substitution）。
    -   归结产生的后继子句。
    -   最终的证明链回溯，清晰展示如何从初始子句导出空子句。

### 运行原始版本

如果需要，也可以运行原始版本：

```bash
python resolver.py
```
该版本会将所有详细步骤直接打印到控制台。

