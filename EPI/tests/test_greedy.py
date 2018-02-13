from ..greedy import *


# 17.1 Optimize task assignment
# pytest -s EPI\tests\test_greedy.py::test_optimize_task_assignment
def test_optimize_task_assignment():
    res = optimize_task_assignment([10, 9, 8, 1])
    assert res[0].task_1 == 1
    assert res[0].task_2 == 10


# 17.1.1 Maximize profit of tasks
# pytest -s EPI\tests\test_greedy.py::test_maximize_profit_of_tasks
def test_maximize_profit_of_tasks():
    tasks_str = "1,3,5#2,5,6#4,6,5#6,7,4#5,8,11#7,9,2"
    tasks = [Task(int(s.split(',')[0]), int(s.split(',')[1]), int(s.split(',')[2])) for s in tasks_str.split('#')]
    res = maximize_profit_of_tasks(tasks)
    assert res == 17
