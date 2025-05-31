import torch
from datasets import load_dataset
from dcscore_function import DCScore


def score(ds_name):
    device = "cuda" if torch.cuda.is_available() else "CPU"
    batch_size = 128
    tau = 1
    kernel_type = "cs"

    # evaluated dataset
    ds = load_dataset(ds_name, split="train")
    ds = ds.select(range(100))
    text_list = [d["text"] for d in ds]
    # text_list = ['who are you', 'I am fine', 'good job']

    model_path = "cl-nagoya/ruri-base-v2"
    # dcscore class
    dcscore_evaluator = DCScore(model_path)

    # get embedding
    embeddings, n, d = dcscore_evaluator.get_embedding(text_list, batch_size=batch_size)

    # calculate dcscore based on embedding
    dataset_dcscore = dcscore_evaluator.calculate_dcscore_by_embedding(
        embeddings, kernel_type=kernel_type, tau=tau
    )
    print(f"ds {ds_name}, score: {dataset_dcscore}")


def main():
    score("if001/elementray_l")
    score("izumi-lab/wikinews-ja-20230728")


if __name__ == "__main__":
    main()
