from datetime import datetime
import visdom


# self.vis.vis.get_window_data(env="19-02 11h35")
class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = f"{env_name}_win"

    def plot_loss(self, loss, step, name):
        self.vis.line(
            [loss], [step],
            win=self.loss_win,
            name=name,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean per 10 steps)',
                # showlegend=True,
                # legend=["Train", "Validation"]
            ))
