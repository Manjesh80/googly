from collections import namedtuple


# TODO find combinations so that elements don't overlap for
# (a,b,c,d ) ==> (a,b),(c,d)  OR (a,c)(b,d) etc ...
def find_the_combinations(tasks, size):
    pass


PairedTask = namedtuple('PairedTask', ('task_1', 'task_2'))

Task = namedtuple('Task', ('start', 'end', 'weight'))


# 17.1 Optimize task assignment
def optimize_task_assignment(tasks):
    tasks.sort()
    return [PairedTask(tasks[i], tasks[~i]) for i in range(len(tasks) // 2)]


# 17.1.1 Maximize profit of tasks
def maximize_profit_of_tasks(tasks):
    tasks.sort(key=lambda t: (t.end, t.start))
    result = [t.weight for t in tasks]

    for i in range(len(tasks)):
        max_profit = tasks[i].weight
        prev_max_profit = 0
        for j in range(0, i):
            if tasks[i].start >= tasks[j].end:
                prev_max_profit = max(prev_max_profit, tasks[j].weight)
        tasks[i] = Task(tasks[i].start, tasks[i].end, max_profit + prev_max_profit)
    return max([t.weight for t in tasks])


# 17.8 Largest rectangle under skyline
def calculate_largest_rectangle(heights):
    pillar_indices, max_rectangle_area = [], 0
    for i, h in enumerate(heights + [0]):
        while pillar_indices and heights[pillar_indices[-1] >= h]:
            height = heights[pillar_indices.pop()]
            width = i if not pillar_indices else i - pillar_indices[-1] - 1
            # width = i
            # if pillar_indices:
            #     width = i - pillar_indices[-1] - 1
            max_rectangle_area = max(max_rectangle_area, height * width)
        pillar_indices.append(i)
    return max_rectangle_area


# 17.8 Largest rectangle under skyline
def max_area_under_histogram(histogram):
    stackie = [0]
    global_max_area = 0
    histogram.append(float('-inf'))

    for i in range(1, len(histogram)):
        current_value = histogram[i]
        if not stackie or current_value >= histogram[stackie[-1]]:
            stackie.append(i)
        else:
            while stackie:
                current_top_index = stackie[-1]
                current_top_value = histogram[current_top_index]
                if current_top_value > current_value:
                    stackie.pop()
                    until_index = stackie[-1] if len(stackie) > 0 else -1
                    until_in_histogram = (i - (until_index + 1)) if until_index != -1 else 1
                    # current_area = (i - (until_index + 1)) * current_top_value
                    current_area = until_in_histogram * current_top_value
                    global_max_area = max(global_max_area, current_area)
                else:
                    stackie.append(i)
                    break

    return global_max_area


if __name__ == "__main__":
    # h = [1, 4, 2, 5, 6, 3, 2, 6, 6, 5, 2, 1, 3]
    h = [1, 4, 2, 5]
    h = [2, 1, 1]
    # h = [1, 0, 1, 0, 0]
    res = max_area_under_histogram(h)
    print(res)

# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
# Comment
