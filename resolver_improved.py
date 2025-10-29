from __future__ import annotations

"""
改进版一阶逻辑归结证明器（含合一）：
- 在二元归结前对变量进行 standardize-apart（变量标准化重命名），避免不同子句的同名变量碰撞。
- 引入依赖追踪，导出空子句时回溯整条父子句链，检查是否使用了否定结论子句。

使用方法：
  python resolver_improved.py
将打印两个问题的初始子句、归结轨迹与最终证明链，并标注否定结论是否被使用。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Iterable
import itertools


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
    """变量，用名称区分。"""
    name: str

    def substitute(self, subst: Dict["Variable", Term]) -> Term:
        return subst.get(self, self)

    def variables(self) -> Set["Variable"]:
        return {self}

    def __str__(self) -> str:  # pragma: no cover
        return self.name


@dataclass(frozen=True)
class Constant(Term):
    """常量，表示具体个体（如 j = John）。"""
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
    """谓词符号及其实参（术语）。"""
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
    t = apply_subst_term(t, subst)
    if isinstance(t, Variable):
        return t == v
    return v in t.variables()


def apply_subst_term(t: Term, subst: Substitution) -> Term:
    prev: Optional[Term] = None
    cur: Term = t
    while prev != cur:
        prev = cur
        cur = cur.substitute(subst)
    return cur


def unify_terms(a: Term, b: Term, subst: Optional[Substitution] = None) -> Optional[Substitution]:
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
            for xa, ya in zip(x.args, y.args):
                stack.append((xa, ya))
            continue
        return None
    return s


def unify_literals(a: Literal, b: Literal) -> Optional[Substitution]:
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
# Utilities: standardize-apart & formatting
# -----------------------------


_fresh_counter = itertools.count(1)


def clause_variables(c: Clause) -> Set[Variable]:
    vs: Set[Variable] = set()
    for lit in c.literals:
        vs |= lit.variables()
    return vs


def standardize_apart(c: Clause) -> Tuple[Clause, Substitution]:
    """对子句内所有变量生成新鲜变量，返回标准化后的子句与所用替换。"""
    vs = clause_variables(c)
    fresh_id = next(_fresh_counter)
    sigma: Substitution = {}
    for v in vs:
        sigma[v] = Variable(f"{v.name}_{fresh_id}")
    if not sigma:
        return c, {}
    return c.substitute(sigma), sigma


def subst_repr(s: Substitution) -> str:
    if not s:
        return "{}"
    # 稳定输出顺序以提升可读性
    items = []
    for v in sorted(s.keys(), key=lambda vv: vv.name):
        items.append(f"{v}→{s[v]}")
    return "{" + ", ".join(items) + "}"


# -----------------------------
# Resolution engine（归结引擎，含依赖追踪）
# -----------------------------


def resolve(c1: Clause, c2: Clause) -> Iterable[Tuple[Clause, Tuple[int, int], Substitution]]:
    """对两子句尝试所有可能的二元归结。对第二个子句进行 standardize-apart。"""
    c2_std, _ = standardize_apart(c2)
    for i, l1 in enumerate(c1.literals):
        for j, l2 in enumerate(c2_std.literals):
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
            for k, lit in enumerate(c2_std.literals):
                if k == j:
                    continue
                new_literals.append(lit.substitute(s))
            # 去重规范化（基于语法字符串键）
            canon: List[Literal] = []
            seen = set()
            for lit in new_literals:
                key = (lit.negated, lit.pred.name, tuple(map(str, lit.pred.args)))
                if key in seen:
                    continue
                seen.add(key)
                canon.append(lit)
            yield Clause(tuple(canon)), (i, j), s


@dataclass
class ProofEdge:
    parent_i: int
    parent_j: int
    lit_i: int
    lit_j: int
    subst: Substitution


def resolution(
    clauses: Sequence[Clause],
    negated_goal_indices: Set[int],
    log: bool = True,
) -> Tuple[bool, List[str], List[str]]:
    """归结主循环，带依赖追踪：
    - 输入：初始子句列表，以及其中属于“否定结论/目标否定”的索引集合（从1起始）。
    - 输出：(是否证明成功, 逐步轨迹, 回溯的证明链文本)。
    """
    kb: List[Clause] = list(clauses)
    derived_keys: Set[str] = set(map(str, kb))
    trace: List[str] = []

    # 依赖追踪：索引从1开始，与打印一致
    parents: Dict[int, Optional[ProofEdge]] = {i + 1: None for i in range(len(kb))}

    def log_line(s: str) -> None:
        if log:
            trace.append(s)

    def backtrace(empty_idx: int) -> List[str]:
        lines: List[str] = []
        used_negated = False
        visited: Set[int] = set()

        def dfs(i: int) -> None:
            nonlocal used_negated
            if i in visited:
                return
            visited.add(i)
            if i in negated_goal_indices:
                used_negated = True
            pe = parents.get(i)
            if pe is None:
                lines.append(f"C{i}: {kb[i-1]}" + ("  [NEGATED GOAL]" if i in negated_goal_indices else ""))
                return
            dfs(pe.parent_i)
            dfs(pe.parent_j)
            sigma_txt = subst_repr(pe.subst)
            lines.append(
                f"C{i}: resolved from C{pe.parent_i}(lit {pe.lit_i+1}) & C{pe.parent_j}(lit {pe.lit_j+1}) with {sigma_txt} => {kb[i-1]}"
            )

        dfs(empty_idx)
        if not used_negated:
            lines.append("[Warning] Empty clause derived without using any negated goal clause.")
        else:
            lines.append("[OK] Empty clause uses at least one negated goal clause.")
        return lines

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
                    if sres not in derived_keys:
                        kb.append(resolvent)
                        derived_keys.add(sres)
                        new_idx = len(kb)  # 1-based
                        parents[new_idx] = ProofEdge(i + 1, j + 1, i1, i2, subst)
                        changed = True
                    if resolvent.is_empty():
                        empty_idx = len(kb)
                        log_line("Derived empty clause ⊥. Refutation complete.")
                        chain = backtrace(empty_idx)
                        return True, trace, chain
    log_line("No new clauses derivable. Refutation failed.")
    return False, trace, []


# -----------------------------
# Helpers to build CNF problems
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


def howling_hounds_cnf_tagged() -> Tuple[List[Clause], Set[int]]:
    """返回 (子句列表, 否定目标子句的1-based索引集合)。目标反证加入：L(j), HasMouse(j)。"""
    x = V("x")
    j = C("j")  # John
    c1 = clause(P("HasHound", x, neg=True), P("HasHowler", x))
    c2 = clause(P("HasCat", x, neg=True), P("HasMouse", x, neg=True))
    c3 = clause(P("L", x, neg=True), P("HasHowler", x, neg=True))
    c4 = clause(P("HasCat", j), P("HasHound", j))
    g1 = clause(P("L", j))
    g2 = clause(P("HasMouse", j))
    clauses = [c1, c2, c3, c4, g1, g2]
    neg_indices = {5, 6}
    return clauses, neg_indices


def drug_dealer_customs_cnf_tagged() -> Tuple[List[Clause], Set[int]]:
    """返回 (子句列表, 否定结论子句的1-based索引集合)。否定结论：¬CO(y) ∨ ¬DD(y)。"""
    x = V("x")
    y = V("y")
    a = C("a")
    c1 = clause(P("EN", x, neg=True), P("VIP", x), P("CO", F("f", x)))
    c2 = clause(P("EN", x, neg=True), P("VIP", x), P("S", F("f", x), x))
    c3 = clause(P("DD", a))
    c4 = clause(P("EN", a))
    c5 = clause(P("S", y, a, neg=True), P("DD", y))
    c6 = clause(P("DD", x, neg=True), P("VIP", x, neg=True))
    c7 = clause(P("CO", y, neg=True), P("DD", y, neg=True))
    clauses = [c1, c2, c3, c4, c5, c6, c7]
    neg_indices = {7}
    return clauses, neg_indices


def print_resolution(name: str, clauses: List[Clause], negated_goal_indices: Set[int]) -> None:
    print(name)
    print("Knowledge base clauses (CNF):")
    for i, c in enumerate(clauses, 1):
        tag = "  [NEGATED GOAL]" if i in negated_goal_indices else ""
        print(f"  C{i}: {c}{tag}")
    print()
    proved, trace, chain = resolution(clauses, negated_goal_indices, log=True)
    print("Resolution trace:")
    for line in trace:
        print(line)
    print()
    if proved:
        print("Result: Refutation succeeded. Empty clause derived.")
        print("Proof chain (backtrace):")
        for ln in chain:
            print(ln)
    else:
        print("Result: Refutation failed.")


def main() -> None:
    print("== Improved Problem 1: Howling Hounds ==")
    clauses1, neg1 = howling_hounds_cnf_tagged()
    print_resolution("", clauses1, neg1)
    print("\n\n")
    print("== Improved Problem 2: Drug dealer and customs official (Attempt) ==")
    clauses2, neg2 = drug_dealer_customs_cnf_tagged()
    print_resolution("", clauses2, neg2)


if __name__ == "__main__":
    main()


