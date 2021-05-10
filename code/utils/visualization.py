from datetime import datetime
import visdom


# self.vis.vis.get_window_data(env="19-02 11h35")
class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.win_name = {}
        self.win_opts = {}

    def add_wins(self, win_name: list, xlabel: list, ylabel: list, title: list,
                 legend: list):

        for i in range(len(win_name)):
            self.win_name[win_name[i]] = f"{win_name[i]}"
            self.win_opts[win_name[i]] = dict(xlabel=xlabel[i],
                                              ylabel=ylabel[i],
                                              title=title[i],
                                              showlegend=bool(legend[i]),
                                              legend=legend[i])

    def plot_line(self, y, x, name, win_name):
        self.vis.line([y], [x],
                      win=self.win_name[win_name],
                      name=name,
                      update='append',
                      opts=self.win_opts[win_name])
