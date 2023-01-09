
import tensorflow as tf
from tensorflow import saved_model as sm




if __name__ == '__main__':
    #saver = tf.train.Saver()



    with tf.Session() as sess:
        path = '/Users/monarch/Desktop/model_zoo/saved_model/1631818020'
        MetaGraphDef = sm.loader.load(sess, tags=["serve"], export_dir=path)


    #     解析得到 SignatureDef protobuf
    #
        graph = tf.get_default_graph()
        SignatureDef_d = MetaGraphDef.signature_def
        SignatureDef = SignatureDef_d[sm.signature_constants.CLASSIFY_INPUTS]

    #     # 解析得到 3 个变量对应的 TensorInfo protobuf
        X_TensorInfo = SignatureDef.inputs['input_1']
    #     scale_TensorInfo = SignatureDef.inputs['input_2']
    #     y_TensorInfo = SignatureDef.outputs['output']
    #     # 解析得到具体 Tensor
    #     # .get_tensor_from_tensor_info() 函数中可以不传入 graph 参数，TensorFlow 自动使用默认图
    #     X = sm.utils.get_tensor_from_tensor_info(X_TensorInfo, sess.graph)
    #     scale = sm.utils.get_tensor_from_tensor_info(scale_TensorInfo, sess.graph)
    #     y = sm.utils.get_tensor_from_tensor_info(y_TensorInfo, sess.graph)
    #     print(sess.run(scale))
    #     print(sess.run(y, feed_dict={X: [3., 2., 1.]}))

