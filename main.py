from model.main_model import train

if __name__ == '__main__':
    try:
        trainer = train(1, ".\\checkpoints")
    except Exception as e:
        print(f"An error occurred: {e}")
