from torch.utils.tensorboard import SummaryWriter
import wandb


class Logger:
    def __init__(
        self,
        config,
        date_str: str,
        exp_name: str,
        time_str: str,
        seed: int,
        use_wandb=False,
        project: str = "DACBench",
    ):
        log_dir = f"outputs/logs/{date_str}/{exp_name}/{time_str}_seed_{seed}"
        self.use_wandb = use_wandb
        if use_wandb:
            self.run = wandb.init(
                project=project,
                name=f"{date_str}/{exp_name}/{time_str}_seed_{seed}",
                config=config,
            )
            # # define our custom x axis metric
            wandb.define_metric("episode")
            # # set all other train/ metrics to use this step
            wandb.define_metric("Loss/episode", step_metric="episode")
            wandb.define_metric("Reward/episode", step_metric="episode")
            self.eps_cnt = 0
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        if self.use_wandb:
            if "episode" in tag:
                self.eps_cnt += 1
                wandb.log(
                    {
                        "episode": self.eps_cnt,
                        tag: value,
                    }
                )
            else:
                wandb.log({tag: value}, step=step)

    def log_dict(self, dictionary, step):
        for tag, value in dictionary.items():
            self.log_scalar(tag, value, step)

    def close(self):
        self.writer.close()
        if self.use_wandb:
            self.run.finish()
