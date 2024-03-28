import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    from utils import Tester
    # Tester.test_topk()
    # Tester.test_lenet_300_100_muti()
    # Tester.test_lenet_5_muti()
    # Tester.test_lenet_300_100(0.95)
    # Tester.test_lenet_300_100(0.98)
    # Tester.test_lenet_5(0.98)
    # Tester.test_lenet_5(0.99)
    # Tester.test_alexnet(512)
    # Tester.test_masked_vggD()
    # Tester.test_masked_wideres(8)
    # Tester.test_masked_lstm(256)
    Tester.test_masked_gru(256)
    # Tester.test_MaskedVgg_t()
    # Tester.test_loss()
    # Tester.test_train_with_mask(0.99)