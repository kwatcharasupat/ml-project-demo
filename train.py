import hydra
import pytorch_lightning as pl


@hydra.main(config_path="expt")
def main(cfg):
    print(cfg.pretty())

    pl.seed_everything(cfg.seed)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    trainer = pl.Trainer(**cfg.trainer)

    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(f"{hydra_output_dir}/config.yaml", "w") as f:
        f.write(cfg.pretty())

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
