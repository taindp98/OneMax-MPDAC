import sys
import os
import subprocess
from typing import List, Tuple


def run_cmd(cmd):
    p = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    )
    output = p.stdout.decode("utf-8")
    return output, p.returncode


def break_points_to_policy(
    n: int, portfolio: List[int], break_points: List[int]
) -> List[int]:
    """
    Given a portfolio and the break_points, calculate the corresponding policy

    Parameters
    ----------
    n: int
        problem size
    portfolio: List[int]
        a list of radius values in ascending order
    break_points: List[int]
        a list of break_points. Radius portfolio[i] is applied iff break_points[i-1] <= f(x) <= break_points[i]

    Returns
    ----------
    policy: List[int]
        a list of the best radius for each objective value from 0 to n-1.
    """
    # remove radii that are not used (due to duplicated breaking points)
    new_portfolio = []
    new_break_points = []
    for i in range(len(break_points)):
        skip = False
        if i > 0:
            if break_points[i] == break_points[i - 1]:
                skip = True
            else:
                assert break_points[i] > break_points[i - 1]
        if skip:
            continue
        new_portfolio.append(portfolio[i])
        new_break_points.append(break_points[i])

    # parse the optimal policy to an array of radiuses ([0..n-1])
    policy = []
    previous_bp = -1
    for i in range(len(new_portfolio)):
        policy.extend([new_portfolio[i]] * (new_break_points[i] - previous_bp))
        previous_bp = new_break_points[i]

    assert len(policy) == n
    return policy


def save_policy(
    n: int, portfolio: List[int], break_points: List[int], policy_file: str
):
    """
    Save a policy to file, in the format of:
    <n>;<portfolio>;<break_points>

    Parameters
    ----------
    n: int
        problem size
    portfolio: List[int]
        a list of radius values in ascending order
    break_points: List[int]
        a list of break_points. Radius portfolio[i] is applied iff break_points[i-1] <= f(x) <= break_points[i]
    policy_file: str
        path to policy file
    """
    policy_str = f"{n};{','.join([str(r) for r in portfolio])};{','.join([str(p) for p in break_points])}"

    # read currently saved policies
    with open(policy_file, "rt") as f:
        ls_lines = [s[:-1] for s in f.readlines()]

    # if the policy wasn't saved yet, do it
    if policy_str not in ls_lines:
        ls_lines.append(policy_str)
    with open(policy_file, "wt") as f:
        f.write("\n".join(ls_lines))


def improvement_probability(n: int, k: int, i: int) -> float:
    """
    Cumulative probability of improvement as defined in
    Eq. 1 in the paper "Theory-inspired Parameter Control Benchmarks for Dynamic Algorithm Configuration"
    https://arxiv.org/pdf/2202.03259.pdf
    indicating the probability of $k$ bits flipped together with the current fitness level $i$.
    Parameters:
        n (int): The length of the bitstring.
        k (int): The number of bits to flip.
        i (int): The current fitness level, which is the number of leading ones in the bitstring.
    Return:
        Returns a float representing the probability of an improvement.
    """

    ## initial probability is calculated as the ratio of the number of bits to flip to the length of the bitstring.
    probability = k / n
    for j in range(1, k):
        probability *= (n - j - i) / (n - j)
    return probability


def determine_breaking_point(larger: int, smaller: int, n: int) -> int:
    """
    Algorithm 2 in the paper "Theory-inspired Parameter Control Benchmarks for Dynamic Algorithm Configuration"
    https://arxiv.org/pdf/2202.03259.pdf

    Parameters:
        larger (int): The larger mutation strength in the portfolio.
        smaller (int): The smaller mutation strength in the portfolio.
        n (int): The length of the bitstring.
    Return:
        Returns an integer representing the breaking point.
    """
    breaking_point = 0
    for i in range(1, n):
        if improvement_probability(n, larger, i) < improvement_probability(
            n, smaller, i
        ):
            break
        breaking_point = i
    return breaking_point


def determine_optimal_breaking_points(portfolio: List[int], n: int) -> List[int]:
    """
    Parameters:

        portfolio: A list of integers representing mutation strengths, which must be sorted in non-increasing order.
        n: An integer representing the size of the problem (e.g., the length of the bitstring).

    Returns:

        A list of integers representing the optimal breaking points.
    """
    assert all(
        portfolio[i] >= portfolio[i + 1] for i in range(len(portfolio) - 1)
    ), "The portfolio is not sorted."
    assert 1 <= len(portfolio) <= n, "The portfolio size is incorrect."

    portfolio_cardinality = len(portfolio)  ## absolute value of the portfolio |K|
    breaking_points = [-1] * (portfolio_cardinality + 1)
    breaking_points[0] = -1
    breaking_points[portfolio_cardinality] = n - 1
    for i in range(1, portfolio_cardinality):
        ## determine the breaking point between two consecutive values of non-increasing K
        breaking_points[i] = determine_breaking_point(portfolio[i - 1], portfolio[i], n)
    breaking_points = breaking_points[1:]
    return breaking_points


def calculate_optimal_policy(
    n: int, portfolio: List[int], script_dir: str = None
) -> Tuple[List[int]]:
    """
    Calculate the optimal policy for a given portfolio of radii

    Parameters
    ----------
    n: int
        problem size
    portfolio: List[int]
        a list of radius values in ascending order

    Returns
    -------
    break_points: List[int]
        the list of break points (one break point per radius) in the portfolio
    policy: List[int]
        a list of the best radius for each objective value from 0 to n-1.
        For example, given the input n=50 and portfolio=[1 17 33], the output will be [33 17 17 17 17 17 17 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]

    """
    # call Martin's D code to get the optimal break_points (Radius portfolio[i] is applied iff break_points[i-1] <= f(x) <= break_points[i])
    portfolio = sorted(portfolio, reverse=True)
    if script_dir is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
    cmd = f"{script_dir}/calculatePolicy {n} {' '.join([str(x) for x in portfolio])}"
    ## D programming
    # output, rc = run_cmd(cmd)
    # print(f"cmd: {cmd}")
    # assert rc == 0, f"ERROR: fail to run command: {cmd}"
    # assert n > 0, f"ERROR: problem size must be a positive integer"
    # break_points = [
    #     int(s.strip()) for s in output.replace("[", "").replace("]", "").split(",")
    # ]

    ## Python
    break_points = determine_optimal_breaking_points(portfolio, n)
    assert len(break_points) == len(portfolio)

    # convert the calculated break_points to policy
    policy = break_points_to_policy(n, portfolio, break_points)

    return break_points, policy


def main():
    assert (
        len(sys.argv) == 3
    ), "Usage: python calculate_optimal_policy.py <n> <portfolio>. \nExample: python calculate_optimal_policy.py 50 1,17,33"

    # read inputs
    n = int(sys.argv[1])
    portfolio = [int(s.strip()) for s in sys.argv[2].split(",")]

    script_dir = os.path.dirname(os.path.realpath(__file__))

    # calculate optimal policy and print it
    break_points, policy = calculate_optimal_policy(n, portfolio, script_dir)
    print(" ".join([str(x) for x in policy]))

    # save the calculated policy to file
    save_policy(
        n,
        portfolio,
        break_points,
        f"{os.path.dirname(script_dir)}/optimal_policies.txt",
    )


if __name__ == "__main__":
    main()
