from utils import TrainingConfig, Trainer

def main():
    config = TrainingConfig()
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()