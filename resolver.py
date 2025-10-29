from __future__ import annotations

"""
一阶逻辑归结证明器（含合一），内置“第一题：Howling Hounds”示例。

主要功能：
- 术语（变量/常量）、谓词、文字、子句的不可变数据结构
- 合一算法（含 occurs-check 与链式替换扁平化）
- 二元归结（挑选互补文字，做合一并产生后继子句）
- 推理循环（逐对尝试归结，记录详细轨迹，发现空子句即归结反证成功）

使用方法：
  python resolver.py
将打印初始子句、每一步归结与替换信息、以及最终是否导出空子句（完成反证）。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Iterable


# -----------------------------
# Term / Predicate definitions
# -----------------------------


class Term:
    """术语基类：变量与常量的共同接口。"""
    def substitute(self, subst: Dict["Variable", "Term"]) -> "Term":
        raise NotImplementedError

    def variables(self) -> Set["Variable"]:
        raise NotImplementedError


@dataclass(frozen=True)
class Variable(Term):
    """变量，用名称区分。

    注：使用不可变数据结构，便于作为字典键；替换时按捕获自替换表。
    """
    name: str

    def substitute(self, subst: Dict["Variable", Term]) -> Term:
        return subst.get(self, self)

    def variables(self) -> Set["Variable"]:
        return {self}

    def __str__(self) -> str:  # pragma: no cover
        return self.name


@dataclass(frozen=True)
class Constant(Term):
    """常量，表示具体个体（如 j = John）。

    常量不受替换影响，变量集合为空。
    """
    name: str

    def substitute(self, subst: Dict[Variable, Term]) -> Term:
        return self

    def variables(self) -> Set[Variable]:
        return set()

    def __str__(self) -> str:  # pragma: no cover
        return self.name


@dataclass(frozen=True)
class Function(Term):
    """函数项（用于Skolem函数等）：name(args...)."""
    name: str
    args: Tuple[Term, ...]

    def substitute(self, subst: Dict[Variable, Term]) -> Term:
        return Function(self.name, tuple(arg.substitute(subst) for arg in self.args))

    def variables(self) -> Set[Variable]:
        vs: Set[Variable] = set()
        for a in self.args:
            vs |= a.variables()
        return vs

    def __str__(self) -> str:  # pragma: no cover
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(map(str, self.args))})"


@dataclass(frozen=True)
class Predicate:
    """谓词符号及其实参（术语）。

    注意：本实验未引入函数符号，仅变量/常量作为参数。
    """
    name: str
    args: Tuple[Term, ...]

    def substitute(self, subst: Dict[Variable, Term]) -> "Predicate":
        return Predicate(self.name, tuple(arg.substitute(subst) for arg in self.args))

    def variables(self) -> Set[Variable]:
        vs: Set[Variable] = set()
        for a in self.args:
            vs |= a.variables()
        return vs

    def __str__(self) -> str:  # pragma: no cover
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(map(str, self.args))})"


@dataclass(frozen=True)
class Literal:
    """文字：谓词或其否定。"""
    pred: Predicate
    negated: bool = False

    def negate(self) -> "Literal":
        return Literal(self.pred, not self.negated)

    def substitute(self, subst: Dict[Variable, Term]) -> "Literal":
        return Literal(self.pred.substitute(subst), self.negated)

    def variables(self) -> Set[Variable]:
        return self.pred.variables()

    def key(self) -> Tuple[str, int]:
        return (self.pred.name, len(self.pred.args))

    def complementary(self, other: "Literal") -> bool:
        return self.pred.name == other.pred.name and len(self.pred.args) == len(other.pred.args) and self.negated != other.negated

    def __str__(self) -> str:  # pragma: no cover
        return ("¬" if self.negated else "") + str(self.pred)


@dataclass(frozen=True)
class Clause:
    """子句：若干文字的析取。空子句表示矛盾（⊥）。"""
    literals: Tuple[Literal, ...]

    def substitute(self, subst: Dict[Variable, Term]) -> "Clause":
        return Clause(tuple(l.substitute(subst) for l in self.literals))

    def is_empty(self) -> bool:
        return len(self.literals) == 0

    def __str__(self) -> str:  # pragma: no cover
        if not self.literals:
            return "⊥"
        return " ∨ ".join(map(str, self.literals))


# -----------------------------
# Unification（合一）
# -----------------------------


Substitution = Dict[Variable, Term]


def occurs_check(v: Variable, t: Term, subst: Substitution) -> bool:
    """occurs-check：避免 v 被替换为含有 v 的项，导致无限递归。

    先应用已有替换再检查，确保正确性。
    """
    t = apply_subst_term(t, subst)
    if isinstance(t, Variable):
        return t == v
    return v in t.variables()


def apply_subst_term(t: Term, subst: Substitution) -> Term:
    """应用替换，按“直到不再变化”为止，扁平化链式替换。"""
    prev: Optional[Term] = None
    cur: Term = t
    while prev != cur:
        prev = cur
        cur = cur.substitute(subst)
    return cur


def unify_terms(a: Term, b: Term, subst: Optional[Substitution] = None) -> Optional[Substitution]:
    """术语合一：返回最一般合一（MGU）或 None（失败）。

    仅支持变量/常量；常量不同即失败；变量采用赋值并进行 occurs-check。
    """
    s: Substitution = {} if subst is None else dict(subst)
    stack: List[Tuple[Term, Term]] = [(a, b)]
    while stack:
        x, y = stack.pop()
        x = apply_subst_term(x, s)
        y = apply_subst_term(y, s)
        if x == y:
            continue
        if isinstance(x, Variable):
            if occurs_check(x, y, s):
                return None
            s[x] = y
            continue
        if isinstance(y, Variable):
            if occurs_check(y, x, s):
                return None
            s[y] = x
            continue
        if isinstance(x, Constant) and isinstance(y, Constant):
            if x != y:
                return None
            continue
        if isinstance(x, Function) and isinstance(y, Function):
            if x.name != y.name or len(x.args) != len(y.args):
                return None
            # 分解为参数对的合一
            for xa, ya in zip(x.args, y.args):
                stack.append((xa, ya))
            continue
        # 其他类型不匹配
        return None
    return s


def unify_literals(a: Literal, b: Literal) -> Optional[Substitution]:
    """文字合一：仅在谓词名/元数相同，且一正一负时尝试合一参数。"""
    if a.pred.name != b.pred.name or len(a.pred.args) != len(b.pred.args):
        return None
    if a.negated == b.negated:
        return None
    s: Substitution = {}
    for ta, tb in zip(a.pred.args, b.pred.args):
        s = unify_terms(ta, tb, s)
        if s is None:
            return None
    return s


# -----------------------------
# Resolution engine（归结引擎）
# -----------------------------


def resolve(c1: Clause, c2: Clause) -> Iterable[Tuple[Clause, Tuple[int, int], Substitution]]:
    """对两子句尝试所有可能的二元归结，生成（后继子句, 使用的文字索引, 合一替换）。"""
    for i, l1 in enumerate(c1.literals):
        for j, l2 in enumerate(c2.literals):
            if not l1.complementary(l2):
                continue
            s = unify_literals(l1, l2)
            if s is None:
                continue
            new_literals: List[Literal] = []
            for k, lit in enumerate(c1.literals):
                if k == i:
                    continue
                new_literals.append(lit.substitute(s))
            for k, lit in enumerate(c2.literals):
                if k == j:
                    continue
                new_literals.append(lit.substitute(s))
            # 替换后可能出现语法重复的文字，这里做一次去重规范化
            canon = []
            seen = set()
            for lit in new_literals:
                key = (lit.negated, lit.pred.name, tuple(map(str, lit.pred.args)))
                if key in seen:
                    continue
                seen.add(key)
                canon.append(lit)
            yield Clause(tuple(canon)), (i, j), s


def resolution(clauses: Sequence[Clause], log: bool = True) -> Tuple[bool, List[str]]:
    """归结主循环：
    - 逐对取子句做归结，新增后继子句加入知识库
    - 若导出空子句（⊥），则完成反证，返回 True
    - 若不再产生新子句，则失败返回 False
    同时收集详细日志（参与子句、替换、结果）。
    """
    kb: List[Clause] = list(clauses)
    derived: Set[str] = set(map(str, kb))
    trace: List[str] = []

    def log_line(s: str) -> None:
        if log:
            trace.append(s)

    changed = True
    step = 1
    while changed:
        changed = False
        n = len(kb)
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = kb[i], kb[j]
                for resolvent, (i1, i2), subst in resolve(c1, c2):
                    line = (
                        f"Step {step}: resolve [C{i+1}:{c1}] (lit {i1+1}) with [C{j+1}:{c2}] (lit {i2+1})\n"
                        f"         substitution: {subst_repr(subst)}\n"
                        f"         => {resolvent}"
                    )
                    log_line(line)
                    step += 1
                    sres = str(resolvent)
                    if sres not in derived:
                        kb.append(resolvent)
                        derived.add(sres)
                        changed = True
                    if resolvent.is_empty():
                        log_line("Derived empty clause ⊥. Refutation complete.")
                        return True, trace
    log_line("No new clauses derivable. Refutation failed.")
    return False, trace


def subst_repr(s: Substitution) -> str:
    """将替换表格式化打印，形如 {x→j, y→a}。"""
    if not s:
        return "{}"
    items = []
    for v, t in s.items():
        items.append(f"{v}→{t}")
    return "{" + ", ".join(items) + "}"


# -----------------------------
# Problem 1: Howling Hounds（问题一编码）
# -----------------------------


def V(name: str) -> Variable:
    return Variable(name)


def C(name: str) -> Constant:
    return Constant(name)


def F(name: str, *args: Term) -> Function:
    return Function(name, tuple(args))


def P(name: str, *args: Term, neg: bool = False) -> Literal:
    return Literal(Predicate(name, tuple(args)), negated=neg)


def clause(*lits: Literal) -> Clause:
    return Clause(tuple(lits))


def howling_hounds_cnf() -> List[Clause]:
    """将题目一的知识与目标否定转为CNF子句集。

    谓词：HasHound/HasCat/HasMouse/HasHowler/L
    个体：j（John）
    目标：证明 L(j) → ¬HasMouse(j)，反证加入 L(j), HasMouse(j)
    """
    x = V("x")
    j = C("j")  # John
    # 1) ∀x (HasHound(x) → HasHowler(x))  => ¬HasHound(x) ∨ HasHowler(x)
    c1 = clause(P("HasHound", x, neg=True), P("HasHowler", x))
    # 2) ∀x (HasCat(x) → ¬HasMouse(x)) => ¬HasCat(x) ∨ ¬HasMouse(x)
    c2 = clause(P("HasCat", x, neg=True), P("HasMouse", x, neg=True))
    # 3) ∀x (L(x) → ¬HasHowler(x)) => ¬L(x) ∨ ¬HasHowler(x)
    c3 = clause(P("L", x, neg=True), P("HasHowler", x, neg=True))
    # 4) HasCat(j) ∨ HasHound(j)
    c4 = clause(P("HasCat", j), P("HasHound", j))
    # Negated goal: L(j) ∧ HasMouse(j)
    g1 = clause(P("L", j))
    g2 = clause(P("HasMouse", j))
    return [c1, c2, c3, c4, g1, g2]


# -----------------------------
# Problem 2: Drug dealer and customs official（尝试版）
"""问题二编码说明（中文）：

谓词与含义：
- EN(x): x 进入了国家（entered the country）
- VIP(x): x 是 VIP
- CO(y): y 是海关官员（customs official）
- S(y,x): y 搜查了 x（searched）
- DD(x): x 是毒贩（drug dealer）

目标（要证）：从 (1)(2)(3) 推出 ∃y (CO(y) ∧ DD(y))。
方法：反证法，加入否定结论 ∀y (CO(y) → ¬DD(y))，其CNF为子句：¬CO(y) ∨ ¬DD(y)。

前提的形式化与Skolem化到CNF：
1) “海关搜查所有非VIP的入境者”：
   ∀x (EN(x) ∧ ¬VIP(x) → ∃y (CO(y) ∧ S(y,x)))
   Skolem：y := f(x)
   得到两个子句（拆成CO与S的独立增益）：
   - ¬EN(x) ∨ VIP(x) ∨ CO(f(x))
   - ¬EN(x) ∨ VIP(x) ∨ S(f(x), x)

2) “存在一些毒贩入境，且他们只被毒贩搜查”：
   ∃x (DD(x) ∧ EN(x) ∧ ∀y (S(y,x) → DD(y)))
   Skolem 常元 a：
   - DD(a)
   - EN(a)
   - ∀y (¬S(y,a) ∨ DD(y)) （转CNF后保留为：¬S(y,a) ∨ DD(y)）

3) “没有毒贩是VIP”：
   ∀x (DD(x) → ¬VIP(x))  =>  ¬DD(x) ∨ ¬VIP(x)

否定结论：
   ∀y (CO(y) → ¬DD(y))  =>  ¬CO(y) ∨ ¬DD(y)

注意：此为一次合理的尝试编码，真实课堂版本若对语义有更细要求（如“only searched by drug dealers”的精确定义、海关的全称量化等），可在此基础上细化或调整。
"""
# -----------------------------


def drug_dealer_customs_cnf() -> List[Clause]:
    """将问题二的自然语言前提与否定结论转为CNF子句集（尝试版）。"""
    x = V("x")
    y = V("y")
    a = C("a")  # 存在的毒贩且入境者见证常元
    # 1) 导出：¬EN(x) ∨ VIP(x) ∨ CO(f(x))
    c1 = clause(P("EN", x, neg=True), P("VIP", x), P("CO", F("f", x)))
    # 1) 导出：¬EN(x) ∨ VIP(x) ∨ S(f(x), x)
    c2 = clause(P("EN", x, neg=True), P("VIP", x), P("S", F("f", x), x))
    # 2a) DD(a)
    c3 = clause(P("DD", a))
    # 2b) EN(a)
    c4 = clause(P("EN", a))
    # 2c) ∀y (¬S(y,a) ∨ DD(y))
    c5 = clause(P("S", y, a, neg=True), P("DD", y))
    # 3) ¬DD(x) ∨ ¬VIP(x)
    c6 = clause(P("DD", x, neg=True), P("VIP", x, neg=True))
    # ¬结论) ¬CO(y) ∨ ¬DD(y)
    c7 = clause(P("CO", y, neg=True), P("DD", y, neg=True))
    return [c1, c2, c3, c4, c5, c6, c7]


def main() -> None:
    """入口：先运行问题一，再尝试运行问题二并打印轨迹与结论。"""
    # 问题一
    print("== Problem 1: Howling Hounds ==")
    clauses1 = howling_hounds_cnf()
    print("Knowledge base clauses (CNF):")
    for i, c in enumerate(clauses1, 1):
        print(f"  C{i}: {c}")
    print()
    proved1, trace1 = resolution(clauses1, log=True)
    print("Resolution trace:")
    for line in trace1:
        print(line)
    print()
    if proved1:
        print("Result: L(j) → ¬HasMouse(j) is proved by refutation.")
    else:
        print("Result: Failed to refute. Proof not found.")
    print("\n\n")

    # 问题二（尝试）
    print("== Problem 2: Drug dealer and customs official (Attempt) ==")
    clauses2 = drug_dealer_customs_cnf()
    print("Knowledge base clauses (CNF):")
    for i, c in enumerate(clauses2, 1):
        print(f"  C{i}: {c}")
    print()
    proved2, trace2 = resolution(clauses2, log=True)
    print("Resolution trace:")
    for line in trace2:
        print(line)
    print()
    if proved2:
        print("Result: From (1)(2)(3), ∃y (CO(y) ∧ DD(y)) proved by refutation.")
    else:
        print("Result: Attempt failed to refute. We may refine the encoding.")


if __name__ == "__main__":
    main()


