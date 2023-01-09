

import tensorflow as tf

if __name__ == '__main__':
    def decode_libsvm(line):
        # columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
        # features = dict(zip(CSV_COLUMNS, columns))
        # labels = features.pop(LABEL_COLUMN)
        columns = tf.string_split([line], ' ')
        print("columns",columns)
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        print("labels", labels)
        splits = tf.string_split(columns.values[1:], ':')
        print("splits", splits)
        id_vals = tf.reshape(splits.values, splits.dense_shape)
        print("id_vals", id_vals)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
        print("feat_ids", feat_ids)
        print("feat_vals", feat_vals)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        print("feat_ids", feat_ids)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        print("feat_vals", feat_vals)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels


    def decode_csv(value_column, configMap, decode_col, default_type,decode_col_idx, mode, process=True):
        value_column = tf.strings.regex_replace(value_column, r"\\N", '0')

        values = tf.io.decode_csv(value_column, record_defaults=default_type, field_delim="\001",
                                  select_cols=decode_col_idx)
        features = dict(zip(decode_col, values))
        label = features.pop('label')

        return features, label

    filenames = "/Users/monarch/Desktop/test.txt"
    # dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)
    # print(dataset)
    #
    # dataset2 = tf.data.TextLineDataset("./data/train").map(decode_libsvm, num_parallel_calls=10).prefetch(500000)
    # print(dataset2)

    dataset3 = tf.data.TextLineDataset("./data/train").map(decode_csv, num_parallel_calls=10).prefetch(500000)