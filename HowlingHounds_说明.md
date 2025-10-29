## 实验一：Howling Hounds 归结证明说明

本说明文档对应 `resolver.py` 中实现的一阶逻辑归结证明器与“第一题：Howling Hounds”的形式化与证明过程。

### 一、问题复述

前提（英文原题）：
1. All hounds howl at night
2. Anyone who has any cats will not have any mice
3. Light sleepers do not have anything which howls at night
4. John has either a cat or a hound

需证：If John is a light sleeper, then John does not have any mice

### 二、谓词与常元约定

- HasHound(x): x 有一条猎犬
- HasCat(x): x 有一只猫
- HasMouse(x): x 有老鼠
- HasHowler(x): x 有会在夜里嚎叫的东西
- L(x): x 是浅眠者（light sleeper）
- 常元 j: John

说明：为便于归结，将“拥有某类会嚎叫的东西”抽象为一元谓词 `HasHowler(x)`，从而避免引入二元谓词与存在量词消去带来的 Skolem 化复杂性。

### 三、形式化与CNF化

原公式：
1) ∀x (HasHound(x) → HasHowler(x))
2) ∀x (HasCat(x) → ¬HasMouse(x))
3) ∀x (L(x) → ¬HasHowler(x))
4) HasCat(j) ∨ HasHound(j)

目标：L(j) → ¬HasMouse(j)

采用反证，将目标否定加入知识库：L(j) ∧ HasMouse(j)

化为合取范式（CNF）子句：
- ¬HasHound(x) ∨ HasHowler(x)
- ¬HasCat(x) ∨ ¬HasMouse(x)
- ¬L(x) ∨ ¬HasHowler(x)
- HasCat(j) ∨ HasHound(j)
- L(j)
- HasMouse(j)

以上编码在 `resolver.py` 的 `howling_hounds_cnf()` 函数中。

### 四、证明思路（归结法）

从上述子句集中逐对选取可归结的文字（一正一负、同谓词同元数），进行合一并生成后继子句；若导出空子句（⊥）则反证成功。

一个典型的归结链：
1. (¬HasCat(x) ∨ ¬HasMouse(x)) 与 HasMouse(j) 合一 x:=j，得 ¬HasCat(j)
2. (HasCat(j) ∨ HasHound(j)) 与 ¬HasCat(j) 归结，得 HasHound(j)
3. (¬HasHound(x) ∨ HasHowler(x)) 与 HasHound(j) 合一 x:=j，得 HasHowler(j)
4. (¬L(x) ∨ ¬HasHowler(x)) 与 L(j) 合一 x:=j，得 ¬HasHowler(j)
5. HasHowler(j) 与 ¬HasHowler(j) 归结，导出空子句 ⊥

因此，否定目标不成立，原命题 L(j) → ¬HasMouse(j) 得证。

### 五、程序结构说明

`resolver.py` 主要模块：
- 术语/谓词/文字/子句：不可变数据结构，便于作为集合/字典键值。
- 合一（Unification）：支持变量/常量，含 occurs-check 与链式替换扁平化。
- 归结（Resolution）：尝试所有互补文字对，合一后拼接其余文字并去重。
- 推理循环：逐步扩张子句集，发现空子句立即返回成功；否则直到不再新增子句失败返回。

### 六、运行与输出

在 `Lab1.2` 目录下运行：
```bash
python resolver.py
```

输出包括：
- 初始 CNF 子句列表（标号 C1, C2, ...）
- 每一步归结的参与子句、文字索引、合一替换 σ 与生成子句
- 导出空子句时的提示与最终结论

### 七、扩展与修改建议

- 若需导入自定义问题，可仿照 `howling_hounds_cnf()` 新增构造函数；
- 可加入“启发式选择对”（例如按文字数/新颖度）以减少搜索；
- 可扩展到支持函数符号与Skolem化的完整一阶逻辑输入管线；
- 可增加命令行参数，选择不同内置问题或从文件加载子句集。


