"""
    1. Named Entity Recognition (NER)

    Input: "Tim Cook works at Apple in California"
    Output: [PERSON, PERSON, O, O, ORG, O, LOC]

# Unpack the output
output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

# Now output has shape [batch_size, max_seq_len, hidden_size]
# For each word, make a prediction
word_predictions = self.classifier(output)  # Shape: [batch_size, max_seq_len, num_classes]

    2. Machine Translation

# Get encoder outputs for each input word
encoder_outputs, encoder_hidden = encoder_rnn(packed_input)

# Unpack to use in attention mechanism
unpacked_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)

# For each decoder step, attention looks at different encoder outputs
# to decide which input words to focus on when generating each output word
attention_weights = attention_mechanism(decoder_hidden, unpacked_outputs)

    3. Part-of-Speech Tagging

    Input: "The cat sits on the mat"
    Output: [DET, NOUN, VERB, PREP, DET, NOUN]

# Unpack the output to get hidden states for each word
unpacked_outputs, lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

# Calculate attention scores for each word
attention_scores = self.attention(unpacked_outputs)  # Shape: [batch_size, seq_len, 1]

# Apply attention to get a weighted sum
context = torch.bmm(attention_scores.transpose(1, 2), unpacked_outputs)

# Final prediction using the attention-weighted representation
prediction = self.classifier(context.squeeze(1))
"""
import torch.nn as nn
import torch.nn.functional as F

class BatchCharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_rate=0.2):
        super().__init__()

        # If batch_first=True, then the input and output tensors are provided as
        # (batch, seq, feature) instead of (seq, batch, feature).
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),  # "hidden to output".
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input_seq, seq_lengths):
        """
        input_seq: [batch_size, max_seq_len, input_size].
        seq_lengths: list of actual sequence lengths.

        line_tensor has shape [seq_length, 1, N_LETTERS] (one sequence at a time).
        max_seq_len refers to the length of the longest sequence in the current batch after padding.
        When processing sequences of different lengths in a batch, we need to pad the shorter sequences
        to match the length of the longest one. This creates a rectangular tensor that can be processed
        efficiently by the RNN.

        input_seq has shape [batch_size, max_seq_length, N_LETTERS] (batched sequences).
        See the batch_first argument in the RNN constructor.

        The transformation from original format to this happens in the collate_batch function, where:

        1. We take each line_tensor from the dataset
        2. We remove the middle dimension (the 1) using squeeze
        3. We pad all sequences in a batch to the same length
        4. We stack them into a single tensor where the first dimension is the batch size

        In essence, input_seq is a batched, padded version of multiple line_tensors.

        seq_lengths is a list containing the original length of each sequence in the batch before
        padding.
        For example, if you have three names in a batch:

        "Smith" (length 5)
        "Rodriguez" (length 9)
        "Li" (length 2)

        max_seq_len would be 9 (the length of "Rodriguez"), All sequences would be padded to this
        length.
        Then seq_lengths would be [9, 5, 2] (sorted in descending order).
        input_seq would have shape [3, 9, N_LETTERS] (batch_size=3, max_length=9)

        The lengths are critical for the pack_padded_sequence function,
        which tells the RNN to ignore padding when computing outputs.
        Without this, the RNN would process the padding as if it were actual data,
        which would lead to incorrect results.

        In the collate_batch function, we calculate these lengths with:
        seq_lengths = [seq.size(0) for seq in sequences]
        """
        # Pack padded sequences to ignore padding in computations.
        packed_input = nn.utils.rnn.pack_padded_sequence(
            input_seq, seq_lengths, batch_first=True, enforce_sorted=True
        )

        # Use the final hidden state for classification.
        # hidden shape: [num_layers, batch_size, hidden_size].
        # The final hidden state already contains a "summary" of the entire sequence.
        # You would use packed_output in cases like:
        # Sequence-to-sequence models (e.g., translation)
        # When you need to attend to different positions in the sequence
        # For tasks requiring token-level predictions (like part-of-speech tagging)
        # Essentially, You need to make predictions for EACH POSITION in the sequence
        # You need to use attention mechanisms
        # You want to extract information from INTERMEDIATE states, not just the final state.
        __packed_output, hidden = self.rnn(packed_input)
        hidden = hidden[-1]  # Take the last layer's hidden state.
        output = self.classifier(hidden)

        return output