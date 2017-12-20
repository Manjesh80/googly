from ..common import *

__all__ = ['remove_last_kth_element']


def remove_last_kth_element(ml, k):
    leader = ml
    follower = ml
    leader = move_list_by(leader, k + 1)
    while leader is not None:
        leader = nudge(leader)
        follower = nudge(follower)
    del_node(follower)
    return ml
