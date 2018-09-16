## modified this module (kyungOok)

class Constant:
    ### Data -----------------------------------------------------------------------------------------------------

    VALIDATION_SET_SIZE = 0.08333   #test Divide training data into training and testing data.

    ### Searcher -------------------------------------------------------------------------------------------------
    # MAX_MODEL_NUM:   MAX_MODEL 수 이내에서 모델을 만든다. count 제한
    MAX_MODEL_NUM = 1000    # if len(self.load_searcher().history) >= Constant.MAX_MODEL_NUM: break
    BETA = 2.576            # A float. The beta in the UCB acquisition function.
    KERNEL_LAMBDA = 0.1     # A float. The balance factor in the neural network kernel.
    T_MIN = 0.0001          # A float. The minimum temperature during simulated annealing(가열 냉각, 풀림).
    N_NEIGHBOURS = 8        # def transform(graph)에서 사용: if len(graphs) >= Constant.N_NEIGHBOURS: break
    MAX_MODEL_SIZE = (1 << 25)   # def transform(graph)에서 사용:
                                 # if temp_graph is not None and temp_graph.size() <= Constant.MAX_MODEL_SIZE:  graphs.append(temp_graph)

    ###  Model Defaults ------------------------------------------------------------------------------------------
    DENSE_DROPOUT_RATE   = 0.5  # output_node_id = graph.add_layer(StubDropout(Constant.DENSE_DROPOUT_RATE), output_node_id)
    CONV_DROPOUT_RATE    = 0.25 # output_node_id = graph.add_layer(StubDropout(Constant.CONV_DROPOUT_RATE), output_node_id)
    CONV_BLOCK_DISTANCE  = 2    # CONV_BLOCK size
    DENSE_BLOCK_DISTANCE = 1    # 소스코드에서 사용하지 않는다
    MODEL_LEN = 3               # An integer. Number of convolutional layers in the initial architecture.
    MODEL_WIDTH = 64            # CnnGenerator : output_node_id = graph.add_layer(StubConv(temp_input_channel, model_width, kernel_size=3), output_node_id)
                                # default_model_width: An integer. The number of filters in each layer in the initial architecture.

    ###  ModelTrainer --------------------------------------------------------------------------------------------

    DATA_AUGMENTATION = True    # class ImageSupervised: A boolean value indicating whether the data needs augmentation.
                                # If not define, then it will use the value of Constant.DATA_AUGMENTATION which is True by default.
    MAX_ITER_NUM = 200          # An integer. The maximum number of epochs to train the model.
                                # The training will stop when this number is reached.
    MIN_LOSS_DEC = 1e-4         # class EarlyStop:  def on_epoch_end(self, loss):
                                # if loss > (self.minimum_loss - self._min_loss_dec): self._no_improvement_count += 1
                                # 뭐나면 수행여부를 판단한다.   로스가 감소한다면 계속 진행해야 한다.

    MAX_NO_IMPROVEMENT_NUM = 5  # An integer. The maximum number of epochs when the loss value doesn't decrease.
                                # The training will stop when this number is reached.
    MAX_BATCH_SIZE = 128        #
    LIMIT_MEMORY = False        # if Constant.LIMIT_MEMORY: pass
    SEARCH_MAX_ITER = 200       #
