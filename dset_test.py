# testing pytorch Datasets

from dsdetectron.sequence import SignSequenceDataset
from torchvision.models import resnet18
import torch.nn as nn


dset = SignSequenceDataset(
    "/local/ecw/deepscribe-detectron/data_nov_2020/hotspots_val.json",
    "/local/ecw/deepscribe-detectron/data_nov_2020/signs_nov_2020.csv",
    "/local/ecw/deepscribe-detectron/data_nov_2020/images_cropped",
    top_frequency=10,
)
images, labels, length = dset[0]

print(labels)

# model = resnet18()

# images, labels = dset[0]

# images2, labels2 = dset[1]


# # figuring out how to pad

# lengths = [45, 12]

# padded = nn.utils.rnn.pad_sequence([images[0, :], images2[0, :]], batch_first=True)

# print(padded.size())

# packed = nn.utils.rnn.pack_padded_sequence(padded, lengths, batch_first=True)

# print(packed.data.size())

# output_cnn = model(packed.data)

# # get a new packed sequence

# packed_output_cnn = nn.utils.rnn.PackedSequence(
#     output_cnn,
#     packed.batch_sizes,
#     packed.sorted_indices,
#     packed.unsorted_indices,
# )

# unpacked_output_cnn, _ = nn.utils.rnn.pad_packed_sequence(
#     packed_output_cnn, batch_first=True
# )

# print(unpacked_output_cnn.size())


# print(images.size())

# print(images[0, :].size())

# batch_size, timesteps, C, H, W = images.size()

# c_in = images.view(batch_size * timesteps, C, H, W)

# result = model(c_in)

# r_in = result.view(batch_size, timesteps, -1)

# print(r_in.size())