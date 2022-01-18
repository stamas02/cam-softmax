import numpy as np
import torch
import torch.nn.functional as F


def loss(logits, labels, c, num_classes, scale=20):
    """
    CAM-Loss
    """
    # at this point because of both the features and the weight in the fully connected layer are normalized the
    # logits are basically the cosine of the angle between the feature vector and the corresponding weight vector-
    cos = logits

    # CAM-SOFTMAX
    c_ = -(1 / (np.log2(np.cos(c) + 1) - 1))
    # cos_ is now the output of the cam-softmax activation
    cos_ = (((cos + 1).pow(c_)) / (2 ** (c_ - 1))) - 1

    # turn the labels into a hot representation
    hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    hot_labels = hot_labels.type(torch.cuda.BoolTensor)

    # For the incorrect label leave the original output and for the correct one switch to the cam-softmax output
    output = cos
    output[hot_labels] = cos_[hot_labels]

    return F.cross_entropy(scale * output, labels)
