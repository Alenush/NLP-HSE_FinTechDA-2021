#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM
from torch.optim import Adam

NUM_EPOCHS = 3


class DemoDataset:
    def __init__(self):
        self.samples = torch.load('demo_dataset.pth')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def main():
    parser = ArgumentParser('DDP usage example')
    parser.add_argument('--batch_size', type=int, default=-1)
    args = parser.parse_args()

    # initialize your model
    model = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

    # send your model to GPU
    model.cuda()

    optimizer = Adam(model.parameters(), lr=1e-5)

    # initialize your dataset
    dataset = DemoDataset()

    # initialize Sampler
    sampler = SequentialSampler(dataset)

    # initialize the dataloader
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=args.batch_size
    )

    # put model in train mode
    model.train()

    # start your training!
    for epoch in range(NUM_EPOCHS):
        print('Epoch', epoch)
        for step, batch in enumerate(dataloader):
            # send batch to device
            batch = batch.cuda()

            # forward pass
            output = model(batch, labels=batch)
            loss = output.loss

            # backward pass
            loss.backward()
            optimizer.step()
            if step % 20 == 0:
                print('%03d: Loss %.4f' % (step, loss))


if __name__ == '__main__':
    main()
