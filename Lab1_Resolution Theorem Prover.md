一、实验名称
知识表示与推理-Resolution Theorem Prover    

二、实验目的及要求
1、熟悉Knowledge representation/base的基本概念和表示法；
2、明确命题逻辑和一阶逻辑的运算规则和推理方法；
3、熟练使用Resolution算法自动化推理一阶逻辑问题。

三、实验环境
Python语言/C++语言
四、实验内容
    设计并开发一个基于Two-Pointer resolution的逻辑推理程序，该程序能探索并解决至少以下两个课堂讨论过的一阶逻辑问题：
1.	Howling Hounds
1)	All hounds howl at night
2)	Anyone who has any cats will not have any mice
3)	Light sleepers do not have anything which howls at night
4)	John has either a cat or a hound
Prove: If John is a light sleeper, then John does not have any mice
2.	Drug dealer and customs official
1)	The customs officials searched everyone who entered the country who was not a VIP
2)	Some of the drug dealers entered the country, and they were only searched by drug dealers
3)	No drug dealer was a VIP
4)	Some of the customs officials were drug dealers

提示:
你需要至少考虑以下一些问题：
(1)	如何将一阶逻辑中使用的逻辑符号在程序中表示出来？
(2)	如何设计unification算法？
(3)	如何实现resolution算法中的substitution功能？
(4)	如何设计算法的停机标准？

实验提交：
包含全部的代码和程序（关键的算法与步骤需注释），实验报告.docx，README.txt等。其中README.txt描述你的代码运行环境和方法

五、实验方法与算法设计
详细记录你如何定义输入，逻辑Clauses在你程序中是如何表示的，你的算法原理和停机准则等

六、实验结果
统计并记录总运算步数和最终的输出结果等
附录：
输出你的程序在Resolution过程中每一步的运算内容和结果（包括该步参与运算的clauses,是否发生substitution，以及该单步resolution的结果等）
