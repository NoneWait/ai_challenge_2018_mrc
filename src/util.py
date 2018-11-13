import numpy as np
import tensorflow as tf


def get_record_parser(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        alter_limit =config.alter_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "alternatives": tf.FixedLenFeature([], tf.string),
                                               "feat": tf.FixedLenFeature([], tf.string),
                                               "y": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64),
                                               # "impossible": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(
            features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
            features["ques_idxs"], tf.int32), [ques_limit])
        alternatives = tf.reshape(tf.decode_raw(
            features["alternatives"], tf.int32), [3])
        feat = tf.reshape(tf.decode_raw(
            features["feat"], tf.float32), [ques_limit]
        )
        y = tf.reshape(tf.decode_raw(
            features["y"], tf.float32), [3])
        qa_id = features["id"]
        return context_idxs, ques_idxs,  alternatives, feat, y, qa_id
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs,  alternatives, feat, y, qa_id):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            buckets_min = [np.iinfo(np.int32).min] + buckets
            buckets_max = buckets + [np.iinfo(np.int32).max]
            conditions_c = tf.logical_and(
                tf.less(buckets_min, c_len), tf.less_equal(c_len, buckets_max))
            bucket_id = tf.reduce_min(tf.where(conditions_c))
            return bucket_id

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


def convert_tokens(eval_file, qa_id, yp):
    remapped_dict = {}
    answer_dict = {}
    for qid, p in zip(qa_id, yp):
        labels = eval_file[str(qid)]["labels"]
        query_id = eval_file[str(qid)]["query_id"]
        remapped_dict[qid] = 1 if np.argmax(labels) == p else 0
        if len(eval_file[str(qid)]["alternatives"])==1:
            # print(eval_file[str(qid)]["alternatives"])
            answer_dict[query_id] = eval_file[str(qid)]["alternatives"][0]
        elif len(eval_file[str(qid)]["alternatives"])==2:
            answer_dict[query_id] = eval_file[str(qid)]["alternatives"][0]
        else:
            # print("p", p)
            # print(eval_file[str(qid)]["alternatives"])
            # print("qid", str(qid))
            answer_dict[query_id] = eval_file[str(qid)]["alternatives"][p]
    return remapped_dict, answer_dict


def evaluate(remapped_dict):
    total = 0
    right = 0
    for key in remapped_dict:
        right += remapped_dict[key]
        total += 1

    return {"acc": right/total}


def acc(preds, ground_truths):
    total = 0
    right = 0
    for pred, ground_truth in zip(preds, ground_truths):
        if pred == ground_truth:
            right += 1
        total += 1

    return float(right)/total
