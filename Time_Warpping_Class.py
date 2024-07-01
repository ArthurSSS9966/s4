import numpy as np
from affinewarp import PiecewiseWarping as pw
from affinewarp import ShiftWarping as sw

from utilfun import *


class Time_Warping:
    TW_model = []

    def __init__(self, movement, task, cod):
        self.movement = movement
        self.task = task
        self.cod = cod  # Condition: Left Hand, Right Hand

    def set_model(self, method):
        assert method in ["ShiftWarp", "PiecewiseWarp"]
        self.method = method
        if method == "ShiftWarp":
            self.TW_model = sw(smoothness_reg_scale=20.0)
        elif method == "PiecewiseWarp":
            self.TW_model = pw(
                n_knots=2, warp_reg_scale=1e-6, smoothness_reg_scale=20.0
            )

    def divide_task(self):
        self.task_move = Move_Data_Extract(self.task, self.movement, self.cod)

    def move_fit(self, task_move):
        self.TW_model.fit(task_move, iterations=50, warp_iterations=200)

    def move_transform(self, data):
        self.est_align = self.TW_model.transform(data)

    def get_aligned(self):
        return self.est_align

    def get_shift(self):
        assert self.method == "ShiftWarp"
        return self.TW_model.shifts

    def shift_task(self, taskID):
        shift = self.get_shift()
        stind = 0
        for i in range(0, len(self.task) - 2):
            if self.task[i, -1] == taskID:
                self.task[i, :-2] = self.task[i, :-2] + shift[stind]
                stind = stind + 1


class Time_Warping_Task:
    def __init__(self, movement, task, cod, method):
        self.time_wrap = Time_Warping(movement, task, cod)
        self.method = method

    def timewrap_transform(self):
        self.time_wrap.divide_task()
        self.time_wrap.set_model(method=self.method)
        taskID = set(self.time_wrap.task[:, -1])
        ind = 0
        for ID in taskID:
            self.time_wrap.move_fit(self.time_wrap.task_move[ind])
            self.time_wrap.shift_task(ID)
            ind = ind + 1

    def return_task(self):
        return self.time_wrap.task
