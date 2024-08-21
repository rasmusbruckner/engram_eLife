""" Task class: This class specifies the task variables """


class Task:
    """ This class specifies the instance variables and methods of the object-based memory task """

    def __init__(self):
        """ This function defines the instance variables unique to each instance """

        self.n_days = 21  # number of time points
        self.n_sim = 32  # number of simulations
        self.n_treatments = 4  # number of treatments
        self.reminder_pres_time = 99  # reminder cues presentation time point
        self.test_timepoints = [0, 6, 13, 20]  # retrieval test time points
