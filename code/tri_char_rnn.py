"""
Credit: https://github.com/kensk8er/udacity/blob/master/assignment_6.py#L218
"""

from __future__ import print_function
import os
import numpy as np
import random
import tensorflow as tf

from code.helpers import maybe_download, read_data, CHARACTER_SIZE, BatchGenerator, ngram2id, id2char, id2_ngram
from code.bi_char_rnn import sample_distribution, log_prob, random_distribution
VALID_SIZE = 1000

# model parameters
BATCH_SIZE = 64
NUM_UNROLLINGS = 10
NUM_NODES = 128
NUM_STEPS = 30001
SUMMARY_FREQUENCY = 100
EMBEDDING_DIMENSION = 128
DROPOUT_PROBABILITY = 0.5

VOCAB_SIZE = CHARACTER_SIZE ** 3  # [a-z] + ' ' (trigram)


def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, VOCAB_SIZE], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def trigrams(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible trigrams
    back into its (most likely) trigram representation."""
    return [id2_ngram(bigram_id) for bigram_id in np.argmax(probabilities, 1)]


def main(token_size=3):
    filename = maybe_download('text8.zip', 31344016)
    text = read_data(filename)

    valid_text = text[:VALID_SIZE]
    train_text = text[VALID_SIZE: VALID_SIZE]

    train_batches = BatchGenerator(train_text, BATCH_SIZE, NUM_UNROLLINGS, token_size=token_size, vocab_size=VOCAB_SIZE)
    valid_batches = BatchGenerator(valid_text, 1, 1, token_size=3, vocab_size=VOCAB_SIZE)

    # simple LSTM Model
    graph = tf.Graph()
    with graph.as_default():
        # Parameters for input, forget, cell state, and output gates
        W_lstm = tf.Variable(tf.truncated_normal([EMBEDDING_DIMENSION + NUM_NODES, NUM_NODES * 4]))
        b_lstm = tf.Variable(tf.zeros([1, NUM_NODES * 4]))

        # Variables saving state across unrollings.
        previous_output = tf.Variable(tf.zeros([BATCH_SIZE, NUM_NODES]), trainable=False)
        previous_state = tf.Variable(tf.zeros([BATCH_SIZE, NUM_NODES]), trainable=False)

        # Classifier weights and biases.
        W = tf.Variable(tf.truncated_normal([NUM_NODES, VOCAB_SIZE], -0.1, 0.1))
        b = tf.Variable(tf.zeros([VOCAB_SIZE]))

        # embedding
        embeddings = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_DIMENSION], minval=-1.0, maxval=1.0))

        # Definition of the cell computation.
        def lstm_cell(X, output, state):
            """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            Note that in this formulation, we omit the various connections between the
            previous state and the gates."""
            X_output = tf.concat(1, [X, output])
            all_logits = tf.matmul(X_output, W_lstm) + b_lstm

            input_gate = tf.sigmoid(all_logits[:, :NUM_NODES])
            forget_gate = tf.sigmoid(all_logits[:, NUM_NODES: NUM_NODES * 2])
            output_gate = tf.sigmoid(all_logits[:, NUM_NODES * 2: NUM_NODES * 3])
            temp_state = all_logits[:, NUM_NODES * 3:]
            state = forget_gate * state + input_gate * tf.tanh(temp_state)
            return output_gate * tf.tanh(state), state

        # Input data.
        train_X = list()
        train_labels = list()
        for _ in range(NUM_UNROLLINGS):
            train_X.append(tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1]))
            train_labels.append(tf.placeholder(tf.float32, shape=[BATCH_SIZE, VOCAB_SIZE]))

        # Unrolled LSTM loop.
        outputs = list()
        output = previous_output
        state = previous_state

        for X in train_X:
            embed = tf.reshape(tf.nn.embedding_lookup(embeddings, X), shape=[BATCH_SIZE, -1])
            output, state = lstm_cell(embed, output, state)
            outputs.append(output)

        # State saving across unrollings.
        with tf.control_dependencies([previous_output.assign(output), previous_state.assign(state)]):
            # Classifier.
            logits = tf.nn.xw_plus_b(tf.nn.dropout(tf.concat(0, outputs), DROPOUT_PROBABILITY), W, b)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

        # Optimizer.
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

        # Predictions.
        train_prediction = tf.nn.softmax(logits)

        # Sampling and validation eval: batch 1, no unrolling.
        sample_input = tf.placeholder(tf.int32, shape=[1, 1])
        sample_embed = tf.reshape(tf.nn.embedding_lookup(embeddings, sample_input), shape=[1, -1])
        previous_sample_output = tf.Variable(tf.zeros([1, NUM_NODES]))
        previous_sample_state = tf.Variable(tf.zeros([1, NUM_NODES]))
        reset_sample_state = tf.group(previous_sample_output.assign(tf.zeros([1, NUM_NODES])), previous_sample_state.assign(tf.zeros([1, NUM_NODES])))
        sample_output, sample_state = lstm_cell(sample_embed, previous_sample_output, previous_sample_state)

        with tf.control_dependencies([previous_sample_output.assign(sample_output), previous_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, W, b))


    # Run the model
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        mean_loss = 0

        for step in range(NUM_STEPS):
            batches = train_batches.next()
            feed_dict = dict()

            for batch_id in range(NUM_UNROLLINGS):
                feed_dict[train_X[batch_id]] = np.where(batches[batch_id] == 1)[1].reshape((-1, 1))
                feed_dict[train_labels[batch_id]] = batches[batch_id + 1]

            _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)

            mean_loss += l

            if step % SUMMARY_FREQUENCY == 0:
                if step > 0:
                    mean_loss = mean_loss / SUMMARY_FREQUENCY

                # The mean loss is an estimate of the loss over the last few batches.
                print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))

                mean_loss = 0
                labels = np.concatenate([batch for batch in batches[1:]])
                print('Minibatch perplexity: %.2f' % float(np.exp(log_prob(predictions, labels))))

                if step % (SUMMARY_FREQUENCY * 10) == 0:
                    # Generate some samples.
                    print('=' * 80)

                    for _ in range(5):
                        feed = sample(random_distribution())
                        sentence = trigrams(feed)[0]
                        reset_sample_state.run()

                        for _ in range(79):
                            feed = np.where(feed == 1)[1].reshape((-1, 1))
                            prediction = sample_prediction.eval({sample_input: feed})
                            feed = sample(prediction)
                            sentence += ''.join(trigrams(feed))
                            last_bigram_id = ngram2id(sentence[-2:])
                            feed = np.array([[float(last_bigram_id == bigram_id) for bigram_id in range(VOCAB_SIZE)]])

                        print(sentence)

                    print('=' * 80)

                # Measure validation set perplexity.
                reset_sample_state.run()
                valid_log_prob = 0

                for _ in range(VALID_SIZE):
                    valid_batch = valid_batches.next()
                    predictions = sample_prediction.eval({sample_input: np.where(valid_batch[0] == 1)[1].reshape((-1, 1))})
                    valid_log_prob = valid_log_prob + log_prob(predictions, valid_batch[1])

                print('Validation set perplexity: %.2f' % float(np.exp(valid_log_prob / VALID_SIZE)))


if __name__ == '__main__':
    main()